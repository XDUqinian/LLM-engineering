#include "src/kernels/build_casual_mask.h"

template<typename T>
__global__ void BuildCausalMask(T* mask, int* q_lens, int* k_lens, int max_q_len, int max_k_len)
{
    int tid = threadIdx.x;
    int qlen = q_lens[blockIdx.x];
    int klen = k_lens[blockIdx.x];
    mask += blockIdx.x * max_q_len * max_k_len;
    int offset = threadIdx.x;
    while(offset < max_q_len * max_k_len) {
        int q = offset / max_k_len;
        int k = offset % max_k_len;
        bool is_one = q < qlen && k < klen && k <= q + (klen - qlen) && k >= klen - qlen;
        mask[offset] = static_cast<T>(is_one);

        offset += blockDim.x;
    }
}

template<typename T>
void launchBuildCausalMasks(TensorWrapper<T>* mask, TensorWrapper<int>* q_lens, TensorWrapper<int>* k_lens)
{
    int batch_size = mask->shape[0];
    int max_q_len = mask->shape[1];
    int max_k_len = mask->shape[2];
    BuildCausalMask<<<batch_size, 256>>>(mask->data, q_lens->data, k_lens->data, max_q_len, max_k_len);
}

template void launchBuildCausalMasks(TensorWrapper<float>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens);

template void launchBuildCausalMasks(TensorWrapper<half>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens);