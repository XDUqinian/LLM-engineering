#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/weights/base_weights.h"
#include "src/weights/llama/norm_weights.h"
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"
// residual.shape = [num tokens, hidden_units]
// decoder_out.shape = [num tokens, hidden_units]
template <typename T>
void launchFusedAddBiasResidualRMSNorm(TensorWrapper<T>* residual,
                                       TensorWrapper<T>* decoder_out,
                                       BaseWeight<T>& norm,
                                       T* scale,
                                       float eps);