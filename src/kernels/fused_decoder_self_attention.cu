#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_decoder_self_attention.h"
// kv cache shape = [numlayers, bs, kv head num, max_seq_len, head size]
template <typename T>
__device__ T warpReduceSum(T val) {
    for (int i = 32 / 2; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}
template <typename T>
__device__ T blockReduceSum(T val) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_num = (blockDim.x + 31) / 32;
    __shared__ T warp_sum[64];
    val = warpReduceSum<T>(val);
    if (lane_id == 0) {
        warp_sum[warp_id] = val;
    }
    __syncthreads();
    T sum = tid < warp_num ? warp_sum[tid] : (T)0.0f;
    sum = warpReduceSum<T>(sum);
    return sum;
}

template <typename T>
__device__ T warpReduceMax(T val) {
    for (int i = 32 / 2; i > 0; i >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, i));
    }
    return val;
}
template <typename T>
__device__ T blockReduceMax(T val) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_num = (blockDim.x + 31) / 32;
    __shared__ T warp_max[64];
    val = warpReduceMax(val);
    if (lane_id == 0) {
        warp_max[warp_id] = val;
    }
    __syncthreads();
    T max_val = tid < warp_num ? warp_max[tid] : 0;
    return warpReduceMax<T>(max_val);
}

// block and thread allocation
// 1 block -> head size，后续可改进为1 warp -> 1 head size or 1 block -> multi head size
// 1 grid -> bs * num heads
// 因为自回归，所以每次生成一个 token
// q; input vec [bs, q num heads, 1, head size]
// k; input vec [bs, kv num heads, 1, head size]
// v; input vec [bs, num heads, 1, head size]
// k_cache; output,[bs, kv_head num, max_seq_len, head size] from prompt phase
// v_cache; output,[bs, kv_head num, max_seq_len, head size] from prompt phase
template <typename T>
__global__ void masked_MHA_kernel(
                    T* q,
                    T* k,
                    T* v,
                    T* qkv_bias,
                    T* k_cache,
                    T* v_cache,
                    T* mha_output,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int max_seq_len,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){
    int tid = threadIdx.x;
    int batch_id = blockIdx.x / head_num;
    int head_id = blockIdx.x % head_num;
    int kv_head_id = head_id / (head_num / kv_head_num);

    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = batch_id * batch_stride + head_id * head_stride + tid;
    int k_offset = batch_id * kv_batch_stride+kv_head_id * head_stride + tid;

    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    Vec_t qvec, kvec, vvec;
    int q_offset_vec = batch_id * batch_stride + head_id * head_stride + tid * vec_size;
    int k_offset_vec = batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;
    int cache_offset = batch_id * kv_head_num * max_seq_len * head_size +
                       kv_head_id * max_seq_len * head_size +
                       tid * vec_size;
    int step_stride = head_size; // 在 kv cache 中每一步表示一个token

    const T* q_mem = q;
    const T* k_mem = k;
    const T* v_mem = v;
    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&q_mem[q_offset_vec]));
        kvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&k_mem[k_offset_vec]));
        vvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&v_mem[k_offset_vec]));
    }
    
    // q k smem for block reduce
    // define dynamic smem type is char type!! not T
    // mainly to show how to memory plan dynamic smem 
    // 动态共享内存分配 前head_size*size_of(T)用来保存当前token的q
    // 后面 step*size_of(float) 用来保存q和历史k的注意力得分，顺便缩放一下
    extern __shared__ char sqk[];
    T* sq_scalar = reinterpret_cast<T*>(sqk); // q存在smem的必要性:在第step行把q存进smem，之前的step-1行可以直接从smem load to reg参与计算
    float* logits = reinterpret_cast<float*>(sq_scalar + head_size); // logits存在smem的必要性:所有线程reduce的结果存到logits，需要smem
    Vec_t* sq = reinterpret_cast<Vec_t*>(sq_scalar);
    if (tid * vec_size < head_size) {
        sq[tid] = qvec;
    }
    __syncthreads();
    // 进行 Q*K 的操作, 不对K转置，直接各个位置的元素相乘，最后归约相加
    // q shape = [1, head_size]
    // k shape = [1, head_size]
    // note: convert some constant scalar to vector type for vectorizedly computation
    float scale = rsqrt((float)head_size);
    float zero = 0.0f;
    Vec_t zero_f4 = scalar_cast_vec<Vec_t, T>(zero);
    float4 scale_f4 = scalar_cast_vec<float4, float>(scale);
    for (int iter = 0; iter < step; iter++) {
        // every iter,  q and k vector's shape = [1, head size]
        // reuse k cache
        // TODO: 或许可以在每个step省略掉前step-1的qk dot
        Vec_t kvec_qk = (tid * vec_size < head_size) ? *reinterpret_cast<Vec_t*>(&k_cache[cache_offset + iter * step_stride]) : zero_f4;
        // when final step, update k cache and fetch k from input k vec, rather than kv cache
        if (iter == step - 1 && tid * vec_size < head_size) {
            // TODO: update k cache with k with bias add when model has qkv bias
            // 到最后一个step 时，该 step 是当前token的K，kv cache中没有，但是传进来的指针预留了这部分空间
            // 需要用当前的 k 更新kv cache
            *reinterpret_cast<Vec_t*>(&k_cache[cache_offset + iter * step_stride]) = kvec;
            kvec_qk = kvec;
        }
        Vec_t qk = zero_f4;
        qk.x = (tid * vec_size < head_size) ? sq[tid].x * kvec_qk.x * scale_f4.x : zero;
        qk.y = (tid * vec_size < head_size) ? sq[tid].y * kvec_qk.y * scale_f4.y : zero;
        qk.z = (tid * vec_size < head_size) ? sq[tid].z * kvec_qk.z * scale_f4.z : zero;
        qk.w = (tid * vec_size < head_size) ? sq[tid].w * kvec_qk.w * scale_f4.w : zero;
        T qk_acc = qk.x + qk.y + qk.z + qk.w;   
        //block reduce using multi warp reduce
        //TODO: maybe broadcast the attn score to each thread of the block in blockreducesum
        T attn_score = blockReduceSum<T>(qk_acc);
        if(tid == 0) {
            logits[iter] = attn_score;
        }
        __syncthreads();
    }        
    //softmax(logits), logits.shape = [bs, num heads, 1, step]
    //进行softmax
    T local_logits = tid < step ? (T)logits[tid] : 0;
    __shared__ float row_max, fenmu;
    T block_max = blockReduceMax<T>(local_logits);
    if (tid == 0) {
        row_max = block_max;
    }
    __syncthreads();
    T fenzi = tid < step ? expf(logits[tid] - row_max) : 0;
    T block_fenmu = blockReduceSum<T>(fenzi);
    if (tid == 0) {
        fenmu = block_fenmu + 1e-6;
    }
    __syncthreads();
    if (tid < step) {
        logits[tid] = (T)fenzi / fenmu;
    }
    __syncthreads();
    // logits*V = [bs, num heads, 1, step] * [bs, kv num heads, step, head size]
   if (tid * vec_size < head_size) {
        // 为什么上面那个矩阵乘法不能把这个条件判断外移呢？因为内部需要做reduce
        // note: above if condition is < head size ,not step, because step by step, we have to use [1, step/seqlen] from logits * [1, head size] from v
        // so here we use acc O to acc the one ele logits * one ele v every step iter
        Vec_t O = scalar_cast_vec<Vec_t, T>(0.0f);
        for (int iter = 0; iter < step; iter++) {
            Vec_t vvec_qkv = *reinterpret_cast<Vec_t*>(&v_cache[cache_offset + iter * step_stride]);
            // when final step, update k cache
            if (iter == step-1) {
                // TODO: update k cache with k with bias add
                *reinterpret_cast<Vec_t*>(&v_cache[cache_offset + iter * step_stride]) = vvec;
                // kv cache does not cache cur step kv, so fetch from input v vec of cur step
                vvec_qkv = vvec;        
            }
            O.x += vvec_qkv.x * logits[iter];
            O.y += vvec_qkv.y * logits[iter];
            O.z += vvec_qkv.z * logits[iter];
            O.w += vvec_qkv.w * logits[iter];
        }
        *reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = O;
    }                
}

template<>
__global__ void masked_MHA_kernel(half* q,
                    half* k,
                    half* v,
                    half* qkv_bias,
                    half* k_cache,
                    half* v_cache,
                    half* mha_output,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int max_seq_len,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){// rsqrt(dh)
        // (RussWong) note: to sync with newest fp32 mha
//     int tid = threadIdx.x;
//     //int bid = blockIdx.x;
//     int q_head_id = blockIdx.x;
//     int q_batch_id = blockIdx.y;
//     int kv_head_id = q_head_id / head_num / kv_head_num;
//     int kv_batch_id = q_batch_id;
//     //int q_head_id = bid % head_num;
//     //int q_batch_id = bid / head_num;
//     //int kv_head_id = bid % kv_head_num;
//     //int kv_batch_id = bid / kv_head_num;

//     int batch_stride = head_num * head_size;
//     int kv_batch_stride = kv_head_num * head_size;
//     int head_stride = head_size;
//     int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
//     int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
//     int cache_offset = batch_size * kv_batch_stride;

//     int vec_size = Vec<half>::size;
//     int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
//     int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;
//     half scale = __float2half(rsqrt(float(head_size)));
//     using Vec_t = typename Vec<half>::Type;
//     Vec_t qvec, kvec, vvec;
//     Vec_t scale_vec = scalar_cast_vec<Vec_t>(scale);
//     //reuse q k v reg from rope
//     const half* q_mem = q;
//     const half* k_mem = k;
//     const half* v_mem = v;
//     if (tid * vec_size < head_size) {
//         qvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&q_mem[q_offset_vec]));
//         if (qkv_bias != nullptr){
//             Vec_t q_bias = *reinterpret_cast<Vec_t*>(&qkv_bias[q_head_id * head_size + tid * vec_size]);
//             qvec = __hadd2(qvec, q_bias);
//         }
//         kvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&k_mem[k_offset_vec]));
//         if (qkv_bias != nullptr){
//             Vec_t k_bias =*reinterpret_cast<Vec_t*>(&qkv_bias[kv_head_id * head_size + tid * vec_size + head_num * head_size]);
//             kvec = __hadd2(kvec, k_bias);
//         }
//         //apply_RoPE(qvec, kvec, tid, rotary_embedding_dim, rotary_embedding_base, step);
//         vvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&v_mem[k_offset_vec]));
//         if (qkv_bias != nullptr){
//             Vec_t v_bias =*reinterpret_cast<Vec_t*>(&qkv_bias[kv_head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]);
//             vvec = __hadd2(vvec, v_bias);
//         }
// 	apply_RoPE(qvec, kvec, tid, rotary_embedding_dim, rotary_embedding_base, step);

//     }
//     // q k smem for block reduce
//     extern __shared__ char sqk[];
//     half* sq = reinterpret_cast<half*>(sqk);
//     // half* sk = sq + head_size;
//     // //float* logits = reinterpret_cast<float*>(sk + head_size);
//     // half* sv = sk + head_size;
//     float* logits = reinterpret_cast<float*>(sq + head_size);
//     //sq[tid] = q_mem[qkv_offset];

//     Vec_t* sq_vec = reinterpret_cast<Vec_t*>(sq);
//     // Vec_t* sk_vec = reinterpret_cast<Vec_t*>(sk);
//     // Vec_t* sv_vec = reinterpret_cast<Vec_t*>(sv);
//     if (tid * vec_size < head_size) {
//         // *reinterpret_cast<Vec_t*>(&sq[tid * vec_size]) = qvec;
//         sq_vec[tid] = qvec;
//     }
//     __syncthreads();
//     half zero = (half)0.0f;
//     Vec_t zero_h2 = scalar_cast_vec<Vec_t, half>(zero);
//     Vec_t scale_h2 = scalar_cast_vec<Vec_t, half>(scale);
//     // FT 2.1的写法里面，kv cache是在prompt阶段已经填充，iter=0为token gen的起始iter
//     for(int iter = 0; iter < step; iter++) {
//         // every iter,  q and k's shape = [1, head size]
//         // reuse k cache
//         // float k = k_cache[iter * cache_offset + qkv_offset];
//         //或许可以在每个step省略掉前step-1的qk dot
//         // sk_vec[tid]= *reinterpret_cast<Vec_t*>(&k_cache[iter * cache_offset + k_offset_vec]);
//         // __syncthreads();
//         Vec_t kvec_qk = tid * vec_size < head_size ? *reinterpret_cast<Vec_t*>(&k_cache[iter * cache_offset + k_offset_vec]) : zero_h2;
//         // when final step, update k cache
//         if (iter == step - 1 && tid * vec_size < head_size) {
//             // TODO: update k cache with k with bias add
//             //k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
//             //sk[tid] = k_mem[qkv_offset];
//             *reinterpret_cast<Vec_t*>(&k_cache[iter * cache_offset + k_offset_vec]) = kvec;
//             kvec_qk = kvec;         
//         }

//         // sq[tid] = q_mem[qkv_offset];
//         __syncthreads();
//         Vec_t qk = (tid * vec_size < head_size) ? __hmul2(__hmul2(sq_vec[tid], kvec_qk), scale_h2) : zero_h2;
//         //block reduce using multi warp reduce
//         float qk_fp32 = __half2float(qk.x) + __half2float(qk.y);
//         float attn_score = blockReduceSum<float>(qk_fp32);
//         if(tid == 0) {
//             logits[iter] = attn_score;
// 	    //float q_tmp = (float)(sq_vec[0].x);
// 	    //float k_tmp = (float)(sk_vec[0].x);
// 	    //float scale_tmp = (float)(scale_vec.x);
//             //printf("iter = %d, step=%d, blockIdx.x = %d, in cuda, logits[%d]=%f, qk_fp32 = %f, q_tmp=%f, k_tmp=%f, scale_tmp=%f\n",iter, step, blockIdx.x, iter, logits[iter], qk_fp32, q_tmp, k_tmp, scale_tmp);
// 	}
//         __syncthreads();
//     }
//     //__syncthreads();
//     //softmax(logits), logits.shape = [bs, num heads, 1, step]
//     //if(tid < step){
//     	//printf("logits[%d]=%f\n", tid, logits[tid]);
//     //}
//     float local_logits = tid < step ? logits[tid] : 0;
//     __shared__ float row_max, fenmu;
    
//     float block_max = blockReduceMax<float>(local_logits);
//     if (tid == 0){
//         row_max = block_max;
//     }
//     __syncthreads();
//     float fenzi = tid < step ? expf(local_logits - row_max) : 0;
//     //if(tid < step) {
//     //	printf("after expf, row_max=%f, fenzi=%f, logits=%f\n", row_max, fenzi, local_logits);
//     //}
//     float block_fenmu = blockReduceSum<float>(fenzi);
//     if (tid == 0){
//         fenmu = block_fenmu;
//     }
//     __syncthreads();
//     if(tid < step) {
        
// 	logits[tid] = (float)(fenzi / fenmu);
// //	printf("in cuda, row_max=%f, fenzi=%f, fenmu=%f, logits=%f\n", row_max, fenzi, fenmu, logits[tid]);
	
//     }
//     __syncthreads();

//     // logits*V = [bs, num heads, 1, step] * [max_seq_len or step, bs, num heads, head size]
//     if (tid * vec_size < head_size) {
//         // note: here is head size ,not step, because step by step, we have to use [1, step/seqlen] from logits * [1, head size] from v
//         // so here we use acc O to acc the one ele logits * one ele v every step iter
//         float2 O = scalar_cast_vec<float2>(0.0f);
//         //O.x = 0.0f;
//         //O.y = 0.0f;
//         for(int iter = 0; iter < step; iter++) {
//             // sv_vec[tid]= *reinterpret_cast<Vec_t*>(&v_cache[iter * cache_offset + k_offset_vec]);
//             // __syncthreads();
//             Vec_t vvec_qkv = *reinterpret_cast<Vec_t*>(&v_cache[iter * cache_offset + k_offset_vec]);
//             // when final step, update k cache
//             if (iter == step - 1) {
//                 // TODO: update k cache with k with bias add
//                 // v_cache[iter * cache_offset + k_offset] = v_mem[k_offset];
//                 // sv[tid] = v_mem[k_offset];
//                 *reinterpret_cast<Vec_t*>(&v_cache[iter * cache_offset + k_offset_vec]) = vvec;
//                 vvec_qkv = vvec;  
//             }
// 	    __syncthreads();
//             //if(bid==0 && tid == 0){
//             //printf("when tid=0, v cache = %f\n", sv[tid]);
//             O.x += (logits[iter] * __half2float(vvec_qkv.x));
//             O.y += (logits[iter] * __half2float(vvec_qkv.y));
//             //O += sv[tid] * logits[iter];
//             __syncthreads();
//         }
        
//         // float* mha_output_fp32 = reinterpret_cast<float*>(mha_output);
//         *reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = __float22half2_rn(O);
//     }
}

// qkv_buf shape = [num_tokens, qkv_head_num, head_size] 自回归 [bs, 1, qkv_head_num,head_size]
// kv cache shape = [num layers, bs, kv_head num, max_seq_len, head size] = >[bs, kv_head num, seqlen[history_len: history_len + max q len] , head size]
template <typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,    // qkv gemm 的结果
                            BaseWeight<T>& qkv,           // qkv linear 的 bias
                            TensorWrapper<int>* layer_id, // 第几层
                            TensorWrapper<T>* k_cache,
                            TensorWrapper<T>* v_cache,
                            TensorWrapper<bool>* finished, // 模型生成 token 是否结束
                            TensorWrapper<int>* step,     // 模型生成 token 到第几个了
                            TensorWrapper<T>* mha_output, // attention 的输出 buf
                            LLaMAAttentionStaticParams& static_params){
    const int batch_size = qkv_buf->shape[0]; // 因为自回归，所以每个token属于一个batch
    const int qkv_head_num = qkv_buf->shape[1];
    const int kv_head_num = k_cache->shape[2];
    const int max_seq_len = k_cache->shape[3];
    int head_num = qkv_head_num- 2 * kv_head_num;
    const int head_size = qkv_buf->shape[2];
    const int cur_step = step->getVal();
    const int layer = layer_id->getVal();
    const int layer_offset = layer * batch_size * kv_head_num * head_size;
    size_t smem_size_bytes = head_size * sizeof(T) + cur_step * sizeof(float);//?
    T* qkv_data = qkv_buf->data;
    //qkv_data.shape = [bs, 1, qkv_head_num,head_size]
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;
    T* v = qkv_data + (head_num + kv_head_num) * head_size;

    int rotary_embedding_dim = static_params.rotary_embedding_dim;
    int rotary_embedding_base = static_params.rotary_embedding_base;
    int max_position_embeddings = static_params.max_position_embeddings;
    bool use_dynamic_ntk = static_params.use_dynamic_ntk;
    dim3 grid(head_num * batch_size);
    dim3 block(head_size); // vec size = 4 for fp32
    masked_MHA_kernel<T><<<grid, block, smem_size_bytes>>>(q,
                                                            k,
                                                            v,
                                                            /*(T*)*/qkv.bias,
                                                            k_cache->data + layer_offset,
                                                            v_cache->data + layer_offset,
                                                            mha_output->data,
                                                            batch_size,
                                                            head_num,
                                                            kv_head_num,
                                                            max_seq_len,
                                                            head_size,
                                                            cur_step,
                                                            rotary_embedding_dim,
                                                            rotary_embedding_base);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(mha_output->data, true);
#else
#endif
}

template void launchDecoderMaskedMHA(TensorWrapper<float>* qkv_buf,
                            BaseWeight<float>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<float>* k_cache,
                            TensorWrapper<float>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<float>* mha_output,
                            LLaMAAttentionStaticParams& static_params);

template void launchDecoderMaskedMHA(TensorWrapper<half>* qkv_buf,
                            BaseWeight<half>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<half>* k_cache,
                            TensorWrapper<half>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<half>* mha_output,
                            LLaMAAttentionStaticParams& static_params);
