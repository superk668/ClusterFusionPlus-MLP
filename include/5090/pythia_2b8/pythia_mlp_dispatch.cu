/**
 * ClusterFusion Pythia-2.8B MLP-Only Kernel Dispatch
 * 
 * This file dispatches only the MLP Down projection kernel.
 * The MLP Up + GELU part is assumed to be done by PyTorch.
 * 
 * Input: mlp_intermediate [FFN_DIM] - output of MLP Up + GELU from PyTorch
 * Output: final output after MLP Down + Residual
 */

#include "kernel_mlp.cuh"
#include <torch/extension.h>

torch::Tensor pythia_2b8_mlp_only_sm120(
    torch::Tensor input,              // Original input for residual [1, hidden_dim]
    torch::Tensor attn_output,        // Attention output for residual [1, hidden_dim]  
    torch::Tensor mlp_intermediate,   // MLP Up + GELU output [ffn_dim] from PyTorch
    torch::Tensor mlp_down_weight,    // [hidden_dim, ffn_dim]
    torch::Tensor mlp_down_bias       // [hidden_dim]
) 
{
    cudaFuncSetAttribute(PythiaMlpDownKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    
    uint32_t max_shmem_size = 128 * sizeof(char) + (2 * TMA_LOAD_ONCE * MAX_SMEM_DIM + DIM_PER_BLOCK + 3 * HEAD_DIM) * sizeof(half) + DIM_BLOCK_REDUCE * sizeof(float);
    cudaFuncSetAttribute(PythiaMlpDownKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor output = torch::zeros({1, HIDDEN_DIM}, options);
    
    // Get pointers
    half* o_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* attn_output_ptr = reinterpret_cast<half*>(attn_output.data_ptr<at::Half>());
    half* mlp_intermediate_ptr = reinterpret_cast<half*>(mlp_intermediate.data_ptr<at::Half>());
    half* mlp_down_weight_ptr = reinterpret_cast<half*>(mlp_down_weight.data_ptr<at::Half>());
    half* mlp_down_bias_ptr = reinterpret_cast<half*>(mlp_down_bias.data_ptr<at::Half>());

    // Create TensorMap for MLP down weight
    constexpr uint32_t rank = 2;
    CUtensorMap tensor_map_mlp_down{};
    uint64_t size_mlp_down[rank] = {FFN_DIM, HIDDEN_DIM};
    uint64_t stride_mlp_down[rank - 1] = {FFN_DIM * sizeof(half)};
    uint32_t box_size_mlp_down[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_mlp_down[rank] = {1, 1};
    
    cuTensorMapEncodeTiled(&tensor_map_mlp_down, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, mlp_down_weight_ptr,
        size_mlp_down, stride_mlp_down, box_size_mlp_down, elem_stride_mlp_down, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid(HEAD_NUM * CLUSTER_SIZE);
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    
    // Launch MLP Down kernel
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = max_shmem_size;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    void* kernel_args[] = {
        &o_ptr,
        &input_ptr,
        &attn_output_ptr,
        &mlp_intermediate_ptr,
        &mlp_down_bias_ptr,
        (void*)&tensor_map_mlp_down
    };
    
    cudaLaunchKernelExC(&config, (void*)PythiaMlpDownKernel, kernel_args);
    
    cudaDeviceSynchronize();
    return output;
}

