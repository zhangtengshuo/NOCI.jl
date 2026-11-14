#==============================================================================#
#      ___           ___           ___                                         #
#     /\__\         /\  \         /\  \          ___                           #
#    /::|  |       /::\  \       /::\  \        /\  \                          #
#   /:|:|  |      /:/\:\  \     /:/\:\  \       \:\  \         ___        ___  #
#  /:/|:|  |__   /:/  \:\  \   /:/  \:\  \      /::\__\       /\  \      /\__\ #
# /:/ |:| /\__\ /:/__/ \:\__\ /:/__/ \:\__\  __/:/\/__/       \:\  \    /:/  / #
# \/__|:|/:/  / \:\  \ /:/  / \:\  \  \/__/ /\/:/  /      ___ /::\__\  /:/  /  #
#     |:/:/  /   \:\  /:/  /   \:\  \       \::/__/      /\  /:/\/__/ /:/__/   #
#     |::/  /     \:\/:/  /     \:\  \       \:\__\      \:\/:/  /    \:\  \   #
#     /:/  /       \::/  /       \:\__\       \/__/       \::/  /      \:\  \  #
#     \/__/         \/__/         \/__/               ██   \/__/        \/__/  #
#                                                                              #
#==============================================================================#

# File: src/runtime/cuda_entrypoints.jl
# Purpose: Lazily route explicit GPU requests to the optional CUDA extension.
# SPDX-License-Identifier: Apache-2.0

# GPU entrypoints are resolved only after an explicit GPU request.

"""
    ERIFactorGPUPrecontractionSource(factors, shards, closed)

Construct the evaluation-lifetime owner of validated host ERI factors and their resident CUDA shards.
Factors retain square-root ERI units; construction stores the supplied arrays and lifecycle state, and each device shard must be released through `_close_eri_factors_gpu_precontraction_source!`.
"""
function ERIFactorGPUPrecontractionSource(args...; kwargs...)
    return _cuda_dispatch(:ERIFactorGPUPrecontractionSource, args...; kwargs...)
end

"""
    ERIFactorGPUShard(device, auxiliary_range, factors, lower_labels, upper_labels, closed)

Construct the owner of one auxiliary-column factor shard and its canonical orbital labels on `device`.
The supplied device arrays are retained with square-root ERI units and one-based labels; `_close_eri_factors_gpu_shard!` releases them and marks the shard closed.
"""
function ERIFactorGPUShard(args...; kwargs...)
    return _cuda_dispatch(:ERIFactorGPUShard, args...; kwargs...)
end

"""
    GronorNOCIDirectTwoElectronGPUWorkspace(integrals, one_body_operator, runtime)

Construct a reusable conventional G-NOCI CUDA workspace on `runtime.gpu_device` and upload the packed physical ERIs and one-body operator once.
Scientific inputs are not mutated, device memory is preflighted before allocation, unavailable CUDA execution raises without CPU fallback, and partial allocations are explicitly released on construction failure.
"""
function GronorNOCIDirectTwoElectronGPUWorkspace(args...; kwargs...)
    return _cuda_dispatch(:GronorNOCIDirectTwoElectronGPUWorkspace, args...; kwargs...)
end

"""
    LUCertifiedHybridGPUWorkspace(T, n_active_occupied, batch_width, runtime)

Allocate reusable CUDA inverse, determinant, index and second-cofactor arrays for the hybrid LU worker.
`T` is Float32 or Float64, occupied rank is 1:32 and batch width is positive; runtime settings are read-only, scaled overlaps are dimensionless, and invalid rank/device settings raise `ArgumentError` while allocation failures release partial storage.
"""
function LUCertifiedHybridGPUWorkspace(args...; kwargs...)
    return _cuda_dispatch(:LUCertifiedHybridGPUWorkspace, args...; kwargs...)
end

"""
    LUCertifiedSpinPairGPUWorkspace(T, n_active_occupied, batch_width, runtime)

Allocate reusable LU factors, inverses, complementary cofactors and status arrays for the internal CUDA LU comparator.
The workspace owns dimensionless scaled-overlap buffers on the device selected by `runtime`; occupied rank is 1:32, batch width is positive, invalid settings raise `ArgumentError`, and partial allocations are released on failure.
"""
function LUCertifiedSpinPairGPUWorkspace(args...; kwargs...)
    return _cuda_dispatch(:LUCertifiedSpinPairGPUWorkspace, args...; kwargs...)
end

"""
    PolynomialSpinPairHessianPairGPUWorkspace(T, n_active_occupied, batch_width, runtime)

Allocate the reusable determinant-Hessian pair-space polynomial workspace on the
single CUDA device selected by `runtime`. The runtime is read-only; allocated
arrays are device-owned and must be released with
`_close_polynomial_spin_pair_hessian_pair_gpu_workspace!`.
"""
function PolynomialSpinPairHessianPairGPUWorkspace(args...; kwargs...)
    return _cuda_dispatch(:PolynomialSpinPairHessianPairGPUWorkspace, args...; kwargs...)
end

"""
    _build_lu_certified_hybrid_gpu_worker!(cofactor_scalar, one_body_active, two_body_inactive_active, same_spin_two_body, first_cofactor_values, state_pair_hessian_response, preparation, coefficients, support, feature_offsets, assignment, n_active_occupied, runtime)

Fill the assigned union range of host AHPC intermediates using CPU LU certification and CUDA second-cofactor contraction.
The first five output arrays are mutated, scientific sources are read-only, cofactors are dimensionless and Hamiltonian responses retain energy units; rejected certificates use CPU polynomial evaluation, scratch is memory-bounded, and the returned named tuple records counts, bytes and wall seconds.
"""
function _build_lu_certified_hybrid_gpu_worker!(args...; kwargs...)
    return _cuda_dispatch(:_build_lu_certified_hybrid_gpu_worker!, args...; kwargs...)
end

"""
    _build_lu_certified_spin_pair_gpu_worker!(cofactor_scalar, one_body_active, two_body_inactive_active, same_spin_two_body, first_cofactor_values, state_pair_hessian_response, preparation, coefficients, support, feature_offsets, assignment, n_active_occupied, runtime)

Fill the assigned union range of host AHPC intermediates with the internal all-CUDA LU comparator.
The first five arrays receive dimensionless cofactors and energy-valued responses; source arrays are read-only, rejected certificates use CPU polynomial evaluation, memory limits constrain batch width, and the returned named tuple reports counts, bytes and wall seconds.
"""
function _build_lu_certified_spin_pair_gpu_worker!(args...; kwargs...)
    return _cuda_dispatch(:_build_lu_certified_spin_pair_gpu_worker!, args...; kwargs...)
end

"""
    _build_polynomial_spin_pair_gpu_hessian_pair_worker!(...)

Evaluate one deterministic contiguous union range on one CUDA device using the
state-pair determinant-Hessian response and write directly into disjoint slices
of the final host arrays. Scientific inputs are read-only, no worker-local final
result is retained, and the configured batch-memory bound is preserved.
"""
function _build_polynomial_spin_pair_gpu_hessian_pair_worker!(args...; kwargs...)
    return _cuda_dispatch(:_build_polynomial_spin_pair_gpu_hessian_pair_worker!, args...; kwargs...)
end

"""
    _eri_factors_core_channels_kernel!(inactive_projection, exchange_by_auxiliary, core_projection, n_inactive)

Compute the inactive Coulomb projection and inactive exchange contribution for every ERI-factor auxiliary vector on the GPU.
Inputs are read-only device arrays, both destinations are overwritten in square-root/integral energy units, no allocation occurs, and one CUDA thread handles each auxiliary vector.
"""
function _eri_factors_core_channels_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_eri_factors_core_channels_kernel!, args...; kwargs...)
end

"""
    _eri_factors_inactive_active_exchange_kernel!(exchange_by_auxiliary, left_cross_projection, right_cross_projection, active_rank, n_inactive)

Compute each auxiliary-vector contribution to the inactive-active exchange response `left_cross * right_cross` on the GPU.
Device inputs are read-only, the response-by-auxiliary destination is overwritten in integral energy units, no allocation occurs, and the host caller validates projection dimensions.
"""
function _eri_factors_inactive_active_exchange_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_eri_factors_inactive_active_exchange_kernel!, args...; kwargs...)
end

"""
    _close_eri_factors_gpu_precontraction_source!(source)

Release every device shard owned by an evaluation-lifetime ERI-factor GPU source and mark the source closed.
The operation is idempotent, preserves host factors, mutates only device-resource lifecycle fields, and attempts every shard cleanup before propagating the first CUDA error.
"""
function _close_eri_factors_gpu_precontraction_source!(args...; kwargs...)
    return _cuda_dispatch(:_close_eri_factors_gpu_precontraction_source!, args...; kwargs...)
end

"""
    _close_eri_factors_gpu_shard!(shard)

Release all CUDA allocations owned by one resident ERI-factor shard and mark it closed.
The operation is idempotent, mutates only the shard lifecycle, performs no host scientific allocation, and CUDA cleanup errors propagate.
"""
function _close_eri_factors_gpu_shard!(args...; kwargs...)
    return _cuda_dispatch(:_close_eri_factors_gpu_shard!, args...; kwargs...)
end

"""
    _close_gronor_noci_two_electron_gpu_workspace!(workspace)

Release every device allocation owned by a G-NOCI CUDA workspace and return `nothing`.
The workspace is mutated to an idempotent closed state, scientific host inputs are unaffected, no result is retained on device, and subsequent contraction attempts raise `ArgumentError`.
"""
function _close_gronor_noci_two_electron_gpu_workspace!(args...; kwargs...)
    return _cuda_dispatch(:_close_gronor_noci_two_electron_gpu_workspace!, args...; kwargs...)
end

"""
    _close_lu_certified_hybrid_gpu_workspace!(workspace)

Release inverse, index, determinant and contracted-second buffers owned by the hybrid CUDA workspace and return `nothing`.
The workspace is marked closed after release; repeated calls are no-ops, host scientific inputs are unchanged, and CUDA cleanup failures propagate.
"""
function _close_lu_certified_hybrid_gpu_workspace!(args...; kwargs...)
    return _cuda_dispatch(:_close_lu_certified_hybrid_gpu_workspace!, args...; kwargs...)
end

"""
    _close_lu_certified_spin_pair_gpu_workspace!(workspace)

Release the internal CUDA LU comparator's factors, inverse, cofactor and status buffers and return `nothing`.
The workspace is marked closed after release; repeated calls are no-ops, host scientific inputs are unchanged, and CUDA cleanup failures propagate.
"""
function _close_lu_certified_spin_pair_gpu_workspace!(args...; kwargs...)
    return _cuda_dispatch(:_close_lu_certified_spin_pair_gpu_workspace!, args...; kwargs...)
end

"""
    _close_polynomial_spin_pair_hessian_pair_gpu_workspace!(workspace)

Release every CUDA allocation owned by one determinant-Hessian pair-space
polynomial workspace. The workspace is mutated only by marking it closed after
all arrays have been freed; repeated calls are no-ops.
"""
function _close_polynomial_spin_pair_hessian_pair_gpu_workspace!(args...; kwargs...)
    return _cuda_dispatch(:_close_polynomial_spin_pair_hessian_pair_gpu_workspace!, args...; kwargs...)
end

"""
    _contract_gronor_noci_one_body_gpu!(workspace, preparation; runtime, telemetry=nothing)

Contract one prepared G-NOCI transition density with the device-resident one-body operator and return its bare overlap/electronic contribution.
The reusable CUDA density buffers are overwritten while scientific inputs remain read-only, output follows one-body energy units, device memory is preflighted, and no CPU contraction fallback is permitted.
"""
function _contract_gronor_noci_one_body_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_contract_gronor_noci_one_body_gpu!, args...; kwargs...)
end

"""
    _contract_gronor_noci_two_electron_pair_rows_kernel!(partials, integrals, lower_labels, upper_labels, alpha_factor, beta_factor, pair_sum, left_alpha, left_beta, right_alpha, right_beta, singular)

Contract one canonical first-orbital-pair row per CUDA block and write its reduced physical packed-ERI contribution.
Device inputs are read-only, `partials` is overwritten in ERI energy units, canonical diagonal multiplicities are applied exactly, no allocation occurs, and the launch uses `GRONOR_NOCI_TWO_ELECTRON_GPU_THREADS` threads per block.
"""
function _contract_gronor_noci_two_electron_pair_rows_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_contract_gronor_noci_two_electron_pair_rows_kernel!, args...; kwargs...)
end

"""
    _copy_eri_factors_projection_tile_kernel!(projection, tile_projection, first_response, valid_rows)

Copy valid rows from one device projection tile into the full response-by-auxiliary destination.
The destination is mutated, inputs remain device-resident and read-only, no allocation occurs, and the host caller guarantees bounds and matching auxiliary dimensions.
"""
function _copy_eri_factors_projection_tile_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_copy_eri_factors_projection_tile_kernel!, args...; kwargs...)
end

"""
    _fill_eri_factors_projection_features_kernel!(features, lower_labels, upper_labels, left_basis, right_basis, first_response, tile_width, n_left, n_right)

Fill one device feature tile representing symmetric projections `left_basis' * L_P * right_basis` for packed Cholesky rows.
All arrays are device-resident and dimensionless except the inherited factor projection, the destination is mutated, no allocation occurs inside the kernel, and launch dimensions are validated by the host caller.
"""
function _fill_eri_factors_projection_features_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_fill_eri_factors_projection_features_kernel!, args...; kwargs...)
end

"""
    _fill_gronor_noci_ao_two_electron_factors_gpu!(workspace, preparation)

Build rank-aware AO cofactor matrices and null vectors with reusable single-GPU cuBLAS workspaces for one prepared G-NOCI pair.
The CUDA workspace and its nested host occupied-factor buffer are mutated while the preparation remains read-only, values are dimensionless, no result is copied back to the host, and dimension/device inconsistencies raise before contraction.
"""
function _fill_gronor_noci_ao_two_electron_factors_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_fill_gronor_noci_ao_two_electron_factors_gpu!, args...; kwargs...)
end

"""
    _launch_lu_certified_hybrid_second_cofactor!(workspace, row_pair_indices_device, column_pair_indices_device, state_pair_hessian_response_device, n_active_occupied, width)

Launch one CUDA block per union to contract certified scaled inverses against the energy-valued determinant-Hessian response.
Device indices and responses are read-only, the workspace's scaled second-cofactor output is overwritten, shapes and width must fit the allocated workspace, CUDA launch failures propagate, and the asynchronous launcher returns `nothing`.
"""
function _launch_lu_certified_hybrid_second_cofactor!(args...; kwargs...)
    return _cuda_dispatch(:_launch_lu_certified_hybrid_second_cofactor!, args...; kwargs...)
end

"""
    _launch_lu_certified_spin_pair_gpu!(workspace, scaled_effective_active_overlap_device, row_pair_indices_device, column_pair_indices_device, state_pair_hessian_response_device, n_active_occupied, width)

Launch the internal CUDA LU comparator for one batch of scaled occupied overlaps.
Inputs are read-only device arrays; LU, inverse, cofactor and certificate-status buffers are overwritten, determinant/cofactor values are dimensionless while contracted responses retain energy units, validated dimensions must fit the workspace, and launch failures propagate before returning `nothing`.
"""
function _launch_lu_certified_spin_pair_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_launch_lu_certified_spin_pair_gpu!, args...; kwargs...)
end

"""
    _launch_polynomial_complementary_cofactors_gpu!(buffers, n_active_occupied, width)

Evaluate the determinant, first complementary cofactor, and contracted second
complementary cofactor for occupied ranks 1:32. The retained polynomial arrays
use union-contiguous physical storage. A single direct-power backbone supplies
the Newton coefficients, terminal first-cofactor recurrence, and the exact
symmetry-regrouped occupied-orbital-pair-space second complementary cofactor.
For ranks 2:32 the sparse endpoint uses eight CUDA lanes per union and dense
grouped powers use one 128-thread block per union with a type-aware, fixed-budget
shared-memory stage. Inputs are device buffers, determinant/cofactor outputs and
the regrouping scratch are mutated, and invalid ranks or widths throw before any
kernel launch; the launcher performs no host allocation or scientific fallback.
"""
function _launch_polynomial_complementary_cofactors_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_launch_polynomial_complementary_cofactors_gpu!, args...; kwargs...)
end

"""
    _launch_polynomial_spin_pair_hessian_pair_per_union_gpu!(buffers, n_active_occupied, width)

Evaluate scaled determinants, first cofactors and contracted second cofactors with one CUDA thread per occupied-overlap union.
Device recurrence/output buffers are overwritten using union-contiguous storage; cofactors are dimensionless and contracted responses retain energy units, occupied ranks and widths must fit the buffers, and CUDA launch errors propagate.
"""
function _launch_polynomial_spin_pair_hessian_pair_per_union_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_launch_polynomial_spin_pair_hessian_pair_per_union_gpu!, args...; kwargs...)
end

"""
    _lu_certified_hybrid_batch_width(T, n_active_occupied, n_union, state_pair_hessian_response_bytes, runtime)

Return `(batch_width, available_device_bytes)` for one hybrid LU worker using fixed response storage and per-union inverse storage.
Counts are dimensionless and memory values are bytes; the selected CUDA device is activated, scientific arrays are unchanged, and invalid settings or insufficient memory raise `ArgumentError` or `ResourceLimitError`.
"""
function _lu_certified_hybrid_batch_width(args...; kwargs...)
    return _cuda_dispatch(:_lu_certified_hybrid_batch_width, args...; kwargs...)
end

"""
    _lu_certified_hybrid_second_cofactor_kernel!(...)

Contract the inverse-based second complementary cofactor on CUDA with one block
per union. Pair-pair terms are independent across threads; each thread accumulates
a strided subset and the block reduces to one scaled contracted-second scalar.
The determinant-Hessian response uses the same unordered-pair convention as the
CPU LU-certified path, including the required factor of two.
"""
function _lu_certified_hybrid_second_cofactor_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_lu_certified_hybrid_second_cofactor_kernel!, args...; kwargs...)
end

"""
    _lu_certified_spin_pair_gpu_batch_width(T, n_active_occupied, n_union, n_state_pair_active_responses, runtime)

Return `(batch_width, available_device_bytes)` for the internal CUDA LU comparator under scratch, configured-memory and free-device limits.
Counts are dimensionless and memory values are bytes; the selected CUDA device is activated, scientific arrays are unchanged, and invalid settings or insufficient memory raise `ArgumentError` or `ResourceLimitError`.
"""
function _lu_certified_spin_pair_gpu_batch_width(args...; kwargs...)
    return _cuda_dispatch(:_lu_certified_spin_pair_gpu_batch_width, args...; kwargs...)
end

"""
    _lu_certified_spin_pair_gpu_kernel!(...)

Evaluate one binary-scaled occupied-overlap problem per CUDA thread by partial-
pivoted LU. Row swaps are applied simultaneously to the identity right-hand side;
forward elimination therefore constructs L⁻¹P and backward substitution produces
the inverse without storing a separate pivot vector. The same infinity-norm
residual and condition limits as the CPU LU certificate are enforced. Certified
unions emit determinant, first complementary cofactors, and the contracted second
cofactor in scaled units; rejected unions emit only a status code and are later
recomputed by the polynomial PDDC fallback.
"""
function _lu_certified_spin_pair_gpu_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_lu_certified_spin_pair_gpu_kernel!, args...; kwargs...)
end

"""
    _polynomial_power_terminal_first_complementary_cofactor_gpu!(...)

Evaluate scaled determinants and first cofactors from retained matrix powers and Newton coefficients on CUDA for occupied ranks 1:32.
The recurrence buffers and dimensionless cofactor outputs are mutated; caller-validated device shapes and launch bounds determine storage, and the retained powers are reused by the contracted-second kernels.
"""
function _polynomial_power_terminal_first_complementary_cofactor_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_power_terminal_first_complementary_cofactor_gpu!, args...; kwargs...)
end

"""
    _polynomial_second_complementary_cofactor_blocked_shared_gpu!(...)

Evaluate all dense grouped-power contributions to the contracted second
complementary cofactor with one CUDA block per union. Each small group block
stages `A^p` and its regrouped `W_p` in dynamic shared memory, stages occupied
pairs and Hessian-response IDs in fixed shared memory, and reuses every
energy-valued response load across all staged powers. The second-output row is
incremented after the sparse endpoint kernel; polynomial inputs and response
tables remain read-only. The kernel supports Float32/Float64 ranks 3:32, uses
bounded shared memory selected by the host resource rule, and performs no
allocation, inverse, SVD, rank classification, or fallback.
"""
function _polynomial_second_complementary_cofactor_blocked_shared_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_second_complementary_cofactor_blocked_shared_gpu!, args...; kwargs...)
end

"""
    _polynomial_second_complementary_cofactor_endpoint_gpu!(...)

Evaluate the exact sparse grouped-power-zero endpoint of the contracted second
complementary cofactor for occupied ranks 2:32. Eight CUDA lanes cooperate per
union, overwrite the union's q-by-q regrouping scratch and second-output scalar,
and read all polynomial, Hessian-pair-index, and energy-valued response inputs
without mutation. The kernel allocates only the fixed occupied-pair shared array,
performs no inverse or rank classification, and assumes validated device shapes
from the production launcher.
"""
function _polynomial_second_complementary_cofactor_endpoint_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_second_complementary_cofactor_endpoint_gpu!, args...; kwargs...)
end

"""
    _polynomial_second_complementary_cofactor_regrouped_matrix_gpu!(...)

Construct the exact symmetry-regrouped coefficient-derivative polynomial

`W_p = e_(q-2-2p) A^p + 2 sum_(j=p+1)^(q-2-p)
(-1)^(p+j) e_(q-2-p-j) A^j`

for one grouped power `p` in the reusable q-by-q scratch matrix. The
elementary-symmetric coefficients and retained matrix powers are read-only;
the scratch column for one overlap union is overwritten. Eight CUDA lanes
cooperate on one union, synchronization is warp-local, and the routine
allocates no storage. The formula is rank-uniform for occupied ranks 2:32.
"""
function _polynomial_second_complementary_cofactor_regrouped_matrix_gpu!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_second_complementary_cofactor_regrouped_matrix_gpu!, args...; kwargs...)
end

"""
    _polynomial_spin_pair_gpu_batch_width(T, n_active_occupied, n_union, n_state_pair_active_responses, runtime)

Return `(batch_width, available_device_bytes)` for one polynomial CUDA worker using its fixed response and per-union memory estimates.
Counts are dimensionless and memory values are bytes; selecting the device changes CUDA context, scientific arrays are unchanged, and invalid settings or insufficient memory raise `ArgumentError` or `ResourceLimitError`.
"""
function _polynomial_spin_pair_gpu_batch_width(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_spin_pair_gpu_batch_width, args...; kwargs...)
end

"""
    _polynomial_spin_pair_hessian_pair_gpu_kernel!(...)

Evaluate one scaled occupied-overlap determinant/cofactor problem per CUDA thread using polynomial recurrences and the unordered determinant-Hessian pair response.
Device recurrence and output arrays are overwritten; overlaps and first cofactors are dimensionless, contracted second responses retain energy units, and the host launcher supplies validated shapes and occupied-rank bounds.
"""
function _polynomial_spin_pair_hessian_pair_gpu_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_spin_pair_hessian_pair_gpu_kernel!, args...; kwargs...)
end

"""
    _polynomial_spin_pair_response_ids_to_hessian_pair_indices_kernel!(...)

Transform each uploaded column-major active-response index grid into canonical
column and row determinant-Hessian pair indices using the same
`n_active_occupied^2` UInt16 device storage. The identity
`2*binomial(n_active_occupied,2)+n_active_occupied=n_active_occupied^2` makes the
transformation storage-neutral. One CUDA thread owns one union and scientific
host staging is not mutated.
"""
function _polynomial_spin_pair_response_ids_to_hessian_pair_indices_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_polynomial_spin_pair_response_ids_to_hessian_pair_indices_kernel!, args...; kwargs...)
end

"""
    _precontract_eri_factors_active_hamiltonian_single_gpu(one_body, space, factors, auxiliary_range; runtime, resident_shard=nothing, include_one_body=true, include_exchange_response=true)

Compute one host-owned partial AHPC coefficient set from a contiguous ERI-factor auxiliary-column shard using one selected GPU.
Scientific inputs are read-only, outputs follow one-body and ERI energy units, optional resident factors/labels are borrowed without release, per-device storage is preflighted, and unavailable CUDA execution raises without CPU fallback.
"""
function _precontract_eri_factors_active_hamiltonian_single_gpu(args...; kwargs...)
    return _cuda_dispatch(:_precontract_eri_factors_active_hamiltonian_single_gpu, args...; kwargs...)
end

"""
    _precontract_packed_streamed_active_hamiltonian_gpu(one_body, space, integrals; runtime)

Compute exact conventional `.two` AHPC coefficients by streaming canonical-pair ERI tiles through single-GPU cuBLAS without materializing dense AO Coulomb/exchange superoperators.
Inputs are read-only, outputs remain host-owned in integral energy units, host/device storage and scratch are preflighted, device allocations are explicitly released, and unavailable CUDA execution raises without CPU fallback.
"""
function _precontract_packed_streamed_active_hamiltonian_gpu(args...; kwargs...)
    return _cuda_dispatch(:_precontract_packed_streamed_active_hamiltonian_gpu, args...; kwargs...)
end

"""
    _prepare_eri_factors_gpu_precontraction_source(factors, runtime)

Upload balanced ERI-factor auxiliary shards and canonical labels once for reuse across every AHPC state pair in one matrix evaluation.
Host factors are read-only, device allocation is preflighted and parallelized over the typed device order, partial uploads are released on failure, and the returned source requires explicit cleanup.
"""
function _prepare_eri_factors_gpu_precontraction_source(args...; kwargs...)
    return _cuda_dispatch(:_prepare_eri_factors_gpu_precontraction_source, args...; kwargs...)
end

"""
    _project_eri_factors_gpu(factors_device, lower_labels_device, upper_labels_device, left_basis_device, right_basis_device, tile_width)

Project every packed Cholesky vector into all ordered left/right response products and return a device matrix with shape `(n_left*n_right, n_auxiliary)`.
Inputs are read-only device arrays, output follows square-root ERI units, temporary device storage is explicitly released, CUDA errors propagate, and no host scientific array is materialized.
"""
function _project_eri_factors_gpu(args...; kwargs...)
    return _cuda_dispatch(:_project_eri_factors_gpu, args...; kwargs...)
end

"""
    _project_spin_pair_alpha_first_cofactor_response_gpu_packed!(coulomb_projected_alpha_response, intermediates, kappa_first, kappa_last, coulomb_device, alpha_first_cofactor_response_device, coulomb_projected_alpha_response_device, packed_value_device, feature_offset_device, feature_pair_device)

Upload one contiguous packed alpha first-cofactor slice and compact structural metadata, scatter it directly into reusable CUDA feature storage, project with cuBLAS, and copy only the Coulomb-projected alpha-response block to CPU.
Scientific inputs are read-only, no floating-point reduction is introduced before the existing GEMM, device buffers are reused without CPU fallback, and invalid ranges or undersized buffers raise before launch.
When phase timing is requested, reusable CUDA events measure the packed scatter stage and GEMM on the device timeline and are read after the blocking D2H copy completes; return-copy wall time includes its necessary device wait, avoiding an extra event synchronization without changing scientific arithmetic or queue order.
"""
function _project_spin_pair_alpha_first_cofactor_response_gpu_packed!(args...; kwargs...)
    return _cuda_dispatch(:_project_spin_pair_alpha_first_cofactor_response_gpu_packed!, args...; kwargs...)
end

"""
    _scatter_spin_pair_alpha_first_cofactor_response_kernel!(features, packed_values, feature_offsets, feature_pairs, block_value_first, width, rank)

Scatter one packed alpha-support first-cofactor block into its dense active-response columns on the GPU.
Every CUDA thread owns one output column, packed values and structural masks are read-only, no floating-point reduction occurs, and launch dimensions are supplied by the caller.
"""
function _scatter_spin_pair_alpha_first_cofactor_response_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_scatter_spin_pair_alpha_first_cofactor_response_kernel!, args...; kwargs...)
end

"""
    _union_contiguous_polynomial_recurrence_buffer(buffer)

Borrow a CUDA recurrence allocation with adjacent union IDs contiguous in physical storage and return a transposed indexing wrapper with the original logical dimensions.
The input is a preallocated response-by-batch device matrix; this host-only device-pointer reshape/permutation changes no values, creates no reference-counted CUDA aliases or device storage, retains dimensionless recurrence units, requires all subsequent recurrence reads/writes to use the same wrapper, and must never be applied to uploaded overlaps, index tables, or externally transferred outputs.
"""
function _union_contiguous_polynomial_recurrence_buffer(args...; kwargs...)
    return _cuda_dispatch(:_union_contiguous_polynomial_recurrence_buffer, args...; kwargs...)
end

"""
    _unpack_packed_eri_tile_kernel!(tile, integrals, row_first, column_first, n_rows, n_columns)

Expand one rectangular tile of the symmetric canonical-pair ERI operator from physical packed values on the GPU.
The integral vector is read-only, the leading `n_rows` by `n_columns` tile is overwritten in ERI energy units, no allocation occurs, and launch bounds are supplied by the caller.
"""
function _unpack_packed_eri_tile_kernel!(args...; kwargs...)
    return _cuda_dispatch(:_unpack_packed_eri_tile_kernel!, args...; kwargs...)
end

"""
    _upload_eri_factors_gpu_shard(factors, lower_labels, upper_labels, auxiliary_range, runtime)

Upload one contiguous auxiliary-column shard and the shared canonical labels to the selected CUDA device.
Host inputs are read-only, returned arrays are device-owned and remain live until explicit cleanup, transfer failures release partial allocations, and device/range inconsistencies raise without fallback.
"""
function _upload_eri_factors_gpu_shard(args...; kwargs...)
    return _cuda_dispatch(:_upload_eri_factors_gpu_shard, args...; kwargs...)
end

"""
    build_spin_pair_intermediates_cuda_polynomial_hessian_pair(...)

Build one fixed-occupied-count AHPC spin-pair intermediate domain on exactly one
CUDA device using the state-pair determinant-Hessian unordered-pair response for
the contracted second complementary cofactor. Host active-response staging,
union order, first-cofactor order, binary scaling, batch-width selection, and
floating-point traversal match the validated implementation. Inputs are read-only
and the result is returned in deterministic union-support order.
"""
function build_spin_pair_intermediates_cuda_polynomial_hessian_pair(args...; kwargs...)
    return _cuda_dispatch(:build_spin_pair_intermediates_cuda_polynomial_hessian_pair, args...; kwargs...)
end

"""
    contract_gronor_noci_determinant_pair!(workspace::GronorNOCIDirectTwoElectronGPUWorkspace, preparation, one_body_operator, integrals; runtime, telemetry=nothing)

Return one complete bare conventional G-NOCI determinant-pair `S/H1/H2` value while reusing the supplied CUDA two-electron workspace.
The workspace is mutated and scientific inputs are read-only, Hamiltonian components follow integral energy units, no nuclear/CI weighting occurs, and unavailable or mismatched GPU execution raises without CPU fallback.
"""
function contract_gronor_noci_determinant_pair!(args...; kwargs...)
    return _cuda_dispatch(:contract_gronor_noci_determinant_pair!, args...; kwargs...)
end

"""
    contract_gronor_noci_two_electron!(workspace::GronorNOCIDirectTwoElectronGPUWorkspace, preparation, integrals; runtime, telemetry=nothing)

Contract one prepared conventional G-NOCI determinant pair with reusable device-resident ERIs and AO-factor workspaces.
The workspace and its nested CPU buffers are mutated while scientific inputs remain read-only, output follows ERI energy units, device selection and memory limits are enforced, nullity three or greater returns zero, and no CPU contraction fallback is permitted.
"""
function contract_gronor_noci_two_electron!(args...; kwargs...)
    return _cuda_dispatch(:contract_gronor_noci_two_electron!, args...; kwargs...)
end

"""
    precontract_active_hamiltonian(one_body, space, source::ERIFactorGPUPrecontractionSource; runtime=RuntimeOptions())

Compute AHPC coefficients with evaluation-lifetime resident ERI factor shards, reusing device factors and labels across state pairs.
Inputs are read-only, outputs are host-owned in integral energy units, one task runs per resident GPU in deterministic order, one-body/exchange assembly occurs once, and closed, mismatched, or insufficiently threaded execution raises without fallback.
"""
function precontract_active_hamiltonian(args...; kwargs...)
    return _cuda_dispatch(:precontract_active_hamiltonian, args...; kwargs...)
end

"""
    precontraction_memory(T, space, source::ERIFactorGPUPrecontractionSource; runtime=RuntimeOptions())

Estimate peak host/device storage for one AHPC state-pair contraction while evaluation-lifetime ERI-factor shards are resident on their selected GPUs.
Inputs are read-only, the estimate includes resident factors plus transient projections to enforce the true live device limit, no CUDA operation occurs, and closed or mismatched sources raise.
"""
function precontraction_memory(args...; kwargs...)
    return _cuda_dispatch(:precontraction_memory, args...; kwargs...)
end

"""
    release_julia_motra_gpu_workspace!(workspace)

Release every CUDA array owned by a Julia-MOTRA workspace and reclaim cached device memory.
The workspace becomes unusable, no host scientific input is modified, and release errors from already-freed arrays are ignored during cleanup.
"""
function release_julia_motra_gpu_workspace!(args...; kwargs...)
    return _cuda_dispatch(:release_julia_motra_gpu_workspace!, args...; kwargs...)
end

"""
    transform_ao_eri_factors_gpu!(host_mo, workspace, host_ao, n_vectors)

Upload one AO-factor batch, perform two strided-batched cuBLAS GEMMs, and copy retained-MO matrices back to CPU storage.
The function mutates `host_mo` and device scratch, leaves `host_ao` unchanged, reports CUDA errors directly, and performs no CPU fallback.
"""
function transform_ao_eri_factors_gpu!(args...; kwargs...)
    return _cuda_dispatch(:transform_ao_eri_factors_gpu!, args...; kwargs...)
end
