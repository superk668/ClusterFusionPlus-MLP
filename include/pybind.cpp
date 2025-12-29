#include <torch/extension.h>

// ============================================================================
// MLP-Only Branch for Pythia-2.8B
// This version only accelerates the MLP Down projection
// Attention and MLP Up are handled by PyTorch
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sm120(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    torch::Tensor cos,
    torch::Tensor sin
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sglang_sm120(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor cos,
    torch::Tensor sin
);

void llama_decoder_layer_batch_sglang_sm120(
    torch::Tensor output,
    torch::Tensor residual_output,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor k_cache_ptrs,
    torch::Tensor v_cache_ptrs,
    int layer_id,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor positions,
    torch::Tensor cos_sin
);

// Pythia-2.8B MLP-Only kernel
// Input: mlp_intermediate from PyTorch (after MLP Up + GELU)
// Output: final output after MLP Down + Residual
torch::Tensor pythia_2b8_mlp_only_sm120(
    torch::Tensor input,              // Original input for residual [1, hidden_dim]
    torch::Tensor attn_output,        // Attention output for residual [1, hidden_dim]  
    torch::Tensor mlp_intermediate,   // MLP Up + GELU output [ffn_dim] from PyTorch
    torch::Tensor mlp_down_weight,    // [hidden_dim, ffn_dim]
    torch::Tensor mlp_down_bias       // [hidden_dim]
);

#ifdef COMPILE_SM120
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("llama_decoder_layer", &llama_decoder_layer_sm120, "");
    m.def("llama_decoder_layer_sglang", &llama_decoder_layer_sglang_sm120, "");
    m.def("llama_decoder_layer_batch_decode_sglang", &llama_decoder_layer_batch_sglang_sm120, "");
    
    // Pythia-2.8B MLP-Only
    m.def("pythia_2b8_mlp_only", &pythia_2b8_mlp_only_sm120, "Pythia-2.8B MLP Down projection only");
}
#endif
