#include <iostream>
#include "src/kernels/act_kernel.h"
#include "src/utils/cuda_debug_utils.cuh"
#include "src/utils/macro.h"
template<typename T>
__device__ __forceinline__ T silu(const T& in) {
    return (T)(((float) in) / (1.0f + expf((float) -in)));
}

template<>
__device__ __forceinline__ half2 silu(const half2& in) {
    return make_half2(__float2half(silu<float>((float)in.x)), __float2half(silu<float>((float)in.y)));
}

// 第一个intermediate做silu，结果与第二个intermediate mul
template <typename T>
__global__ void silu_and_mul_kernel(T *out,
                                    T *in,
                                    const int intermedia_size)
{
    int batch_id = blockIdx.x;
    for (int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x)
    {
        T x = in[batch_id * 2 * intermedia_size + idx];
        T y = in[batch_id * 2 * intermedia_size + intermedia_size + idx];
        out[batch_id * intermedia_size + idx] = silu(x)*y;
    }
}

template <>
__global__ void silu_and_mul_kernel(
    half *out,
    half *in,
    const int intermedia_size) {
    int batch_id = blockIdx.x;
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    for(int idx = threadIdx.x * vec_size; idx < intermedia_size; idx += blockDim.x) {
        Vec_t x = *reinterpret_cast<Vec_t*>(in + batch_id * 2 * intermedia_size + idx);
        Vec_t y = *reinterpret_cast<Vec_t*>(in + batch_id * 2 * intermedia_size + intermedia_size + idx);
        *reinterpret_cast<Vec_t*>(out + batch_id * intermedia_size + idx) = __hmul2(silu<Vec_t>(x), y);
    }
}

// out [bs, intermedia size]
// in  [bs, 2, intermedia size]
template <typename T>
void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out) {
    int batch_size = input->shape[0];
    LLM_CHECK(input->shape[1] == 2);
    int intermedia_size = input->shape[2];
    dim3 grid(batch_size);
    dim3 block(256);
    silu_and_mul_kernel<T><<<grid, block>>>(out->data, input->data, intermedia_size);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(out->data);
#else
#endif
}

template void launchAct(TensorWrapper<float>* input, TensorWrapper<float>* output);
template void launchAct(TensorWrapper<half>* input, TensorWrapper<half>* output);
