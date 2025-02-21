#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "src/kernels/cal_paddingoffset.h"
int main() {
    const int batch_size = 3;
    const int max_q_len = 5;

    int* h_seq_lens;
    int* d_seq_lens;
    h_seq_lens = (int*)malloc(sizeof(int) * batch_size);
    cudaMalloc((void**)&d_seq_lens, sizeof(int) * batch_size);

    int* h_cum_seqlens;
    int* d_cum_seqlens;
    h_cum_seqlens = (int*)malloc(sizeof(int) * (batch_size + 1));
    cudaMalloc((void**)&d_cum_seqlens, sizeof(int) * (batch_size + 1));

    int* h_padding_offset;
    int* d_padding_offset;
    h_padding_offset = (int*)malloc(sizeof(int) *  batch_size * max_q_len);
    cudaMalloc((void**)&d_padding_offset, sizeof(int) * max_q_len * batch_size);

    for(int i= 0; i < batch_size; i++) {
        h_seq_lens[i] = batch_size;
    }
    cudaMemcpy(d_seq_lens, h_seq_lens, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    DataType type_int = getTensorType<int>();
    TensorWrapper<int>* padding_offset = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, max_q_len}, d_padding_offset);
    TensorWrapper<int>* cum_seqlens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size + 1}, d_cum_seqlens);
    TensorWrapper<int>* input_lengths = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_seq_lens);

    launchCalPaddingoffset(padding_offset, cum_seqlens, input_lengths);

    cudaMemcpy(h_padding_offset, d_padding_offset, sizeof(int) * batch_size * max_q_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cum_seqlens, d_cum_seqlens, sizeof(int) * (batch_size + 1), cudaMemcpyDeviceToHost);

    for(int i = 0; i < batch_size * max_q_len; i++) {
        printf("padding_offset = %d\n", h_padding_offset[i]);
    }
    for(int i = 0; i < batch_size + 1; i++){
        printf("cum_seqlens =%d\n", h_cum_seqlens[i]);
    }

    free(h_seq_lens);
    free(h_padding_offset);
    free(h_cum_seqlens);
    cudaFree(d_seq_lens);
    cudaFree(d_padding_offset);
    cudaFree(d_cum_seqlens);
}