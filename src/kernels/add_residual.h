#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"

// residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
template <typename T>
void launchAddResidual(
    TensorWrapper<T> *residual,
    TensorWrapper<T> *decoder_out, // [num tokens, hidden_units]
    bool is_print=false
);