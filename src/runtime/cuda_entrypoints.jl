# GPU entrypoints are resolved only after an explicit GPU request.

"""
    ERIFactorGPUPrecontractionSource(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function ERIFactorGPUPrecontractionSource(args...; kwargs...)
    return _cuda_dispatch(:ERIFactorGPUPrecontractionSource, args...; kwargs...)
end

"""
    ERIFactorGPUShard(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function ERIFactorGPUShard(args...; kwargs...)
    return _cuda_dispatch(:ERIFactorGPUShard, args...; kwargs...)
end

"""
    DPDirectTwoElectronGPUWorkspace(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function DPDirectTwoElectronGPUWorkspace(args...; kwargs...)
    return _cuda_dispatch(:DPDirectTwoElectronGPUWorkspace, args...; kwargs...)
end

"""
    LUCertifiedHybridGPUWorkspace(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function LUCertifiedHybridGPUWorkspace(args...; kwargs...)
    return _cuda_dispatch(:LUCertifiedHybridGPUWorkspace, args...; kwargs...)
end

"""
    LUCertifiedSpinPairGPUWorkspace(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function LUCertifiedSpinPairGPUWorkspace(args...; kwargs...)
    return _cuda_dispatch(:LUCertifiedSpinPairGPUWorkspace, args...; kwargs...)
end

"""
    PolynomialSpinPairHessianPairGPUWorkspace(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function PolynomialSpinPairHessianPairGPUWorkspace(args...; kwargs...)
    return _cuda_dispatch(:PolynomialSpinPairHessianPairGPUWorkspace, args...; kwargs...)
end

"""
    _build_lu_certified_hybrid_gpu_worker!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _build_lu_certified_hybrid_gpu_worker!(args...; kwargs...)
    return _cuda_dispatch(:_build_lu_certified_hybrid_gpu_worker!, args...; kwargs...)
end

"""
    _build_lu_certified_spin_pair_gpu_worker!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _build_lu_certified_spin_pair_gpu_worker!(args...; kwargs...)
    return _cuda_dispatch(:_build_lu_certified_spin_pair_gpu_worker!, args...; kwargs...)
end

"""
    _build_polynomial_spin_pair_gpu_hessian_pair_worker!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _build_polynomial_spin_pair_gpu_hessian_pair_worker!(args...; kwargs...)
    return _cuda_dispatch(:_build_polynomial_spin_pair_gpu_hessian_pair_worker!, args...; kwargs...)
end

"""
    _eri_factors_core_channels_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _eri_factors_core_channels_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_eri_factors_core_channels_kernel!, args...; kwargs...)
end

"""
    _eri_factors_inactive_active_exchange_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _eri_factors_inactive_active_exchange_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_eri_factors_inactive_active_exchange_kernel!, args...; kwargs...)
end

"""
    _close_eri_factors_gpu_precontraction_source!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _close_eri_factors_gpu_precontraction_source!(args...; kwargs...)
    return _cuda_dispatch(:_close_eri_factors_gpu_precontraction_source!, args...; kwargs...)
end

"""
    _close_eri_factors_gpu_shard!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _close_eri_factors_gpu_shard!(args...; kwargs...)
    return _cuda_dispatch(:_close_eri_factors_gpu_shard!, args...; kwargs...)
end

"""
    _close_dp_two_electron_gpu_workspace!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _close_dp_two_electron_gpu_workspace!(args...; kwargs...)
    return _cuda_dispatch(:_close_dp_two_electron_gpu_workspace!, args...; kwargs...)
end

"""
    _close_lu_certified_hybrid_gpu_workspace!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _close_lu_certified_hybrid_gpu_workspace!(args...; kwargs...)
    return _cuda_dispatch(:_close_lu_certified_hybrid_gpu_workspace!, args...; kwargs...)
end

"""
    _close_lu_certified_spin_pair_gpu_workspace!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _close_lu_certified_spin_pair_gpu_workspace!(args...; kwargs...)
    return _cuda_dispatch(:_close_lu_certified_spin_pair_gpu_workspace!, args...; kwargs...)
end

"""
    _close_polynomial_spin_pair_hessian_pair_gpu_workspace!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _close_polynomial_spin_pair_hessian_pair_gpu_workspace!(args...; kwargs...)
    return _cuda_dispatch(:_close_polynomial_spin_pair_hessian_pair_gpu_workspace!, args...; kwargs...)
end

"""
    _contract_dp_one_body_gpu!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _contract_dp_one_body_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_contract_dp_one_body_gpu!, args...; kwargs...)
end

"""
    _contract_dp_two_electron_pair_rows_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _contract_dp_two_electron_pair_rows_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_contract_dp_two_electron_pair_rows_kernel!, args...; kwargs...)
end

"""
    _copy_eri_factors_projection_tile_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _copy_eri_factors_projection_tile_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_copy_eri_factors_projection_tile_kernel!, args...; kwargs...)
end

"""
    _fill_eri_factors_projection_features_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _fill_eri_factors_projection_features_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_fill_eri_factors_projection_features_kernel!, args...; kwargs...)
end

"""
    _fill_dp_ao_two_electron_factors_gpu!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _fill_dp_ao_two_electron_factors_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_fill_dp_ao_two_electron_factors_gpu!, args...; kwargs...)
end

"""
    _launch_lu_certified_hybrid_second_cofactor!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _launch_lu_certified_hybrid_second_cofactor!(args...; kwargs...)
    return _cuda_dispatch(:_launch_lu_certified_hybrid_second_cofactor!, args...; kwargs...)
end

"""
    _launch_lu_certified_spin_pair_gpu!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _launch_lu_certified_spin_pair_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_launch_lu_certified_spin_pair_gpu!, args...; kwargs...)
end

"""
    _launch_polynomial_complementary_cofactors_gpu!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _launch_polynomial_complementary_cofactors_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_launch_polynomial_complementary_cofactors_gpu!, args...; kwargs...)
end

"""
    _launch_polynomial_spin_pair_hessian_pair_per_union_gpu!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _launch_polynomial_spin_pair_hessian_pair_per_union_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_launch_polynomial_spin_pair_hessian_pair_per_union_gpu!, args...; kwargs...)
end

"""
    _lu_certified_hybrid_batch_width(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _lu_certified_hybrid_batch_width(args...; kwargs...)
    return _cuda_dispatch(:_lu_certified_hybrid_batch_width, args...; kwargs...)
end

"""
    _lu_certified_hybrid_second_cofactor_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _lu_certified_hybrid_second_cofactor_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_lu_certified_hybrid_second_cofactor_kernel!, args...; kwargs...)
end

"""
    _lu_certified_spin_pair_gpu_batch_width(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _lu_certified_spin_pair_gpu_batch_width(args...; kwargs...)
    return _cuda_dispatch(:_lu_certified_spin_pair_gpu_batch_width, args...; kwargs...)
end

"""
    _lu_certified_spin_pair_gpu_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _lu_certified_spin_pair_gpu_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_lu_certified_spin_pair_gpu_kernel!, args...; kwargs...)
end

"""
    _polynomial_power_terminal_first_complementary_cofactor_gpu!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _polynomial_power_terminal_first_complementary_cofactor_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_power_terminal_first_complementary_cofactor_gpu!, args...; kwargs...)
end

"""
    _polynomial_second_complementary_cofactor_blocked_shared_gpu!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _polynomial_second_complementary_cofactor_blocked_shared_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_second_complementary_cofactor_blocked_shared_gpu!, args...; kwargs...)
end

"""
    _polynomial_second_complementary_cofactor_endpoint_gpu!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _polynomial_second_complementary_cofactor_endpoint_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_second_complementary_cofactor_endpoint_gpu!, args...; kwargs...)
end

"""
    _polynomial_second_complementary_cofactor_regrouped_matrix_gpu!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _polynomial_second_complementary_cofactor_regrouped_matrix_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_second_complementary_cofactor_regrouped_matrix_gpu!, args...; kwargs...)
end

"""
    _polynomial_spin_pair_gpu_batch_width(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _polynomial_spin_pair_gpu_batch_width(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_spin_pair_gpu_batch_width, args...; kwargs...)
end

"""
    _polynomial_spin_pair_hessian_pair_gpu_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _polynomial_spin_pair_hessian_pair_gpu_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_spin_pair_hessian_pair_gpu_kernel!, args...; kwargs...)
end

"""
    _polynomial_spin_pair_response_ids_to_hessian_pair_indices_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _polynomial_spin_pair_response_ids_to_hessian_pair_indices_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_spin_pair_response_ids_to_hessian_pair_indices_kernel!, args...; kwargs...)
end

"""
    _precontract_eri_factors_active_hamiltonian_single_gpu(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _precontract_eri_factors_active_hamiltonian_single_gpu(args...; kwargs...)
    return _cuda_dispatch(:_precontract_eri_factors_active_hamiltonian_single_gpu, args...; kwargs...)
end

"""
    _precontract_packed_streamed_active_hamiltonian_gpu(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _precontract_packed_streamed_active_hamiltonian_gpu(args...; kwargs...)
    return _cuda_dispatch(:_precontract_packed_streamed_active_hamiltonian_gpu, args...; kwargs...)
end

"""
    _prepare_eri_factors_gpu_precontraction_source(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _prepare_eri_factors_gpu_precontraction_source(args...; kwargs...)
    return _cuda_dispatch(:_prepare_eri_factors_gpu_precontraction_source, args...; kwargs...)
end

"""
    _project_eri_factors_gpu(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _project_eri_factors_gpu(args...; kwargs...)
    return _cuda_dispatch(:_project_eri_factors_gpu, args...; kwargs...)
end

"""
    _project_spin_pair_alpha_first_cofactor_response_gpu_packed!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _project_spin_pair_alpha_first_cofactor_response_gpu_packed!(args...; kwargs...)
    return _cuda_dispatch(:_project_spin_pair_alpha_first_cofactor_response_gpu_packed!, args...; kwargs...)
end

"""
    _scatter_spin_pair_alpha_first_cofactor_response_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _scatter_spin_pair_alpha_first_cofactor_response_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_scatter_spin_pair_alpha_first_cofactor_response_kernel!, args...; kwargs...)
end

"""
    _union_contiguous_polynomial_recurrence_buffer(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _union_contiguous_polynomial_recurrence_buffer(args...; kwargs...)
    return _cuda_dispatch(:_union_contiguous_polynomial_recurrence_buffer, args...; kwargs...)
end

"""
    _unpack_packed_eri_tile_kernel!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _unpack_packed_eri_tile_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_unpack_packed_eri_tile_kernel!, args...; kwargs...)
end

"""
    _upload_eri_factors_gpu_shard(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function _upload_eri_factors_gpu_shard(args...; kwargs...)
    return _cuda_dispatch(:_upload_eri_factors_gpu_shard, args...; kwargs...)
end

"""
    build_spin_pair_intermediates_cuda_polynomial_hessian_pair(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function build_spin_pair_intermediates_cuda_polynomial_hessian_pair(args...; kwargs...)
    return _cuda_dispatch(:build_spin_pair_intermediates_cuda_polynomial_hessian_pair, args...; kwargs...)
end

"""
    contract_dp_determinant_pair!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function contract_dp_determinant_pair!(args...; kwargs...)
    return _cuda_dispatch(:contract_dp_determinant_pair!, args...; kwargs...)
end

"""
    contract_dp_two_electron!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function contract_dp_two_electron!(args...; kwargs...)
    return _cuda_dispatch(:contract_dp_two_electron!, args...; kwargs...)
end

"""
    precontract_active_hamiltonian(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function precontract_active_hamiltonian(args...; kwargs...)
    return _cuda_dispatch(:precontract_active_hamiltonian, args...; kwargs...)
end

"""
    precontraction_memory(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function precontraction_memory(args...; kwargs...)
    return _cuda_dispatch(:precontraction_memory, args...; kwargs...)
end

"""
    release_julia_motra_gpu_workspace!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function release_julia_motra_gpu_workspace!(args...; kwargs...)
    return _cuda_dispatch(:release_julia_motra_gpu_workspace!, args...; kwargs...)
end

"""
    transform_ao_eri_factors_gpu!(args...; kwargs...)

Forward an explicit GPU operation to the optional CUDA extension.
Arguments and return values follow the concrete extension method; units and mutation are preserved, missing optional support raises `GPUUnavailableError`, and CPU execution does not invoke this hook.
"""
function transform_ao_eri_factors_gpu!(args...; kwargs...)
    return _cuda_dispatch(:transform_ao_eri_factors_gpu!, args...; kwargs...)
end
