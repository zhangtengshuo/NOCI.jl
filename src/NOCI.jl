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

#=
 NOCI.jl — Nonorthogonal Configuration Interaction in Julia

 File:            src/NOCI.jl
 Purpose:         Top-level module including every source file in dependency order.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

module NOCI

using Dates
using LinearAlgebra
using Printf
import HDF5
import JLD2
import SHA
import YAML
import TOML

include("runtime/version.jl")

include("core/types.jl")
include("core/results.jl")
include("integrals/backends.jl")
include("runtime/telemetry.jl")
include("runtime/optional_cuda.jl")
include("runtime/openmolcas4noci.jl")
include("integrals/values.jl")
include("integrals/fortran_records.jl")
include("algorithms/ahpc_noci/options.jl")
include("integral_sources/MolcasIO.jl")
include("integral_sources/SewardShards.jl")
include("integral_sources/JuliaMotra.jl")
include("core/solver.jl")
include("runtime/production_types.jl")
include("runtime/atomic_io.jl")
include("runtime/progress.jl")
include("core/correlation_shifts.jl")
include("core/couplings.jl")
include("integrals/factor_provenance.jl")
include("integrals/loaders.jl")
include("integral_sources/SewardHDF5.jl")
include("integrals/seward_runtime.jl")
include("core/wavefunctions.jl")
include("core/spin_manifolds.jl")
include("core/input.jl")
include("core/mebf.jl")
include("core/multifragment_mebf.jl")
include("core/multifragment_mebf_parallel_fill.jl")
include("core/multifragment_mebf_collection.jl")
include("core/multifragment_mebf_plan.jl")
include("core/multifragment_mebf_production.jl")
include("core/multifragment_count_preflight.jl")
include("core/pairs.jl")
include("core/matrix_utils.jl")
include("runtime/checkpoint.jl")
include("algorithms/ahpc_noci/complementary_cofactors.jl")
include("algorithms/ahpc_noci/polynomial_cofactors.jl")
include("algorithms/ahpc_noci/polynomial_second_contraction.jl")
include("algorithms/ahpc_noci/polynomial_workspace.jl")
include("algorithms/ahpc_noci/ordinary_polynomial_workspace.jl")
include("algorithms/ahpc_noci/precontraction.jl")
include("runtime/cuda_resources/precontraction_cuda.jl")
include("algorithms/ahpc_noci/state_pair.jl")
include("algorithms/ahpc_noci/precontraction_source.jl")
include("algorithms/ahpc_noci/spin_support.jl")
include("algorithms/ahpc_noci/spin_weights.jl")
include("algorithms/ahpc_noci/spin_incidence.jl")
include("algorithms/ahpc_noci/spin_result.jl")
include("algorithms/ahpc_noci/spin_intermediates.jl")
include("algorithms/ahpc_noci/spin_contraction.jl")
include("algorithms/ahpc_noci/hessian_pair_response.jl")
include("algorithms/ahpc_noci/state_pair_hessian_response.jl")
include("algorithms/ahpc_noci/lu_certified_cofactors.jl")
include("algorithms/ahpc_noci/implicit_spin_features.jl")
include("algorithms/ahpc_noci/implicit_spin_intermediate_construction.jl")
include("runtime/cuda_resources/polynomial_spin_pair_cuda_support.jl")
include("runtime/cuda_resources/polynomial_spin_pair_cuda_runtime.jl")
include("runtime/cuda_resources/polynomial_spin_pair_cuda_trivial.jl")
include("runtime/cuda_resources/polynomial_spin_pair_cuda_hessian_pair.jl")
include("runtime/cuda_resources/polynomial_spin_pair_cuda_multidevice.jl")
include("runtime/cuda_resources/polynomial_spin_pair_cuda_multigpu_hessian_pair.jl")
include("runtime/cuda_resources/polynomial_spin_pair_cuda_multigpu.jl")
include("runtime/cuda_resources/lu_certified_spin_pair_cuda.jl")
include("runtime/cuda_resources/lu_certified_spin_pair_cuda_hybrid.jl")
include("runtime/cuda_resources/polynomial_spin_pair_cuda_complementary_cofactors.jl")
include("algorithms/ahpc_noci/implicit_spin_feature_contraction.jl")
include("algorithms/ahpc_noci/spin_matrix_assembly.jl")
include("algorithms/gronor_noci/pair_preparation.jl")
include("algorithms/gronor_noci/two_electron.jl")
include("runtime/cuda_resources/two_electron_cuda.jl")
include("algorithms/gronor_noci/matrix_assembly.jl")
include("runtime/cuda_entrypoints.jl")
include("runtime/evaluate.jl")
include("runtime/output.jl")
include("runtime/coupling_output.jl")
include("runtime/snapshot.jl")
include("runtime/caspt2_override.jl")
include("runtime/run.jl")

export noci_version
export openmolcas4noci_identity
export NOCIAlgorithm
export GronorNOCI
export AHPCNOCI
export AHPCBranch
export AHPCSpinStringBranch
export AHPCAccelerationOptions
export SupportLookup
export DictPresizedLookup
export FlatHashPresizedLookup
export AdaptiveLookup
export CofactorEvaluation
export PolynomialCofactorEvaluation
export LUCertifiedCofactorEvaluation
export ERIBackend
export PackedTwoERI
export FactorizedERI
export ERIFactorSource
export ERIFactorBundleSource
export SewardMOTRASource
export PackedERIValues
export ERIFactorValues
export ConventionalIntegralLayout
export OneElectronValues
export OneElectronFileData
export ERIFactorBundle
export ERIFactorProvenance
export eri_factor_provenance_dict
export SewardChRst
export SewardShardInfo
export SewardSourceLayout
export JuliaMOTRADimensions
export JuliaMOTRAResult
export SewardHDF5Manifest
export SewardHDF5ImportResult
export SewardMOTRAProduct
export DeterminantExpansion
export PackedActiveOccupations
export DeterminantFileData
export OrbitalCoefficients
export FragmentWavefunction
export ScreeningOptions
export FragmentStateSpecification
export MEBFDefinition
export NOCIInput
export FragmentStateEntry
export MEBFComposition
export NOCIProblemDefinition
export SpinMicrostateExpansion
export MolecularBasisFunction
export DeterminantPairReference
export RowCompressedDeterminantPairs
export DeterminantPairGroup
export DeterminantPairDomain
export OccupiedSpinOrbitals
export PreparedDeterminantPair
export canonical_pair_count
export canonical_pair_index
export packed_eri_from_factors
export packed_eri_from_factors_memory
export detect_fortran_record_marker
export read_fortran_record
export read_one_electron_file
export read_packed_two_eri
export read_eri_factor_bundle
export write_eri_factor_bundle_v2
export inspect_seward_hdf5
export seward_hdf5_import_memory
export import_seward_hdf5
export write_seward_hdf5_products
export inspect_seward_source
export seward_motra_memory
export seward_orbital_dimensions
export transform_seward_eri_factors
export load_seward_eri_factors
export core_hamiltonian
export one_electron_file_memory
export packed_two_eri_memory
export eri_factor_bundle_memory
export orbital_coefficient_memory
export active_orbital_count
export determinant_count
export occupation_string
export read_determinant_file
export read_orbital_coefficients
export read_fragment_wavefunction
export read_noci_input
export problem_definition_memory
export load_problem_definition
export clebsch_gordan_coefficient
export mebf_assembly_memory
export build_molecular_basis_functions
export determinant_pair_domain_memory
export build_determinant_pair_domain
export full_determinant_pair_count
export occupied_orbital_columns
export occupied_spin_orbitals
export prepare_determinant_pair
export UnsupportedERIBackend
export AlgorithmNotImplemented
export ResourceLimitError
export GPUUnavailableError
export RuntimeOptions
export openmolcas4noci_executable
export with_openmolcas4noci
export run_openmolcas4noci
export ahpc_threads_per_gpu
export MemoryEstimate
export MemoryPreflight
export PhaseTelemetry
export ExecutionTelemetry
export RunOutputOptions
export CheckpointOptions
export ProductionRunResult
export Caspt2OverrideResult
export ComplementaryCofactorWorkspace
export ComplementaryCofactorResult
export ActiveResponseSpace
export ActiveHamiltonianCoefficients
export AHPCStatePairPreparation
export RawNOCIMatrices
export NormalizedNOCIMatrices
export NOCISolution
export NOCIEvaluationResult
export CanonicalSpinStringPair
export CanonicalMEBFSpinStrings
export SpinPairSupportDomain
export SparseSpinPairWeights
export PreparedSpinPairIncidence
export AHPCSpinPairMatrixElement
export NonInjectiveSpinPairMapping
export GronorNOCIFullOccupiedCofactors
export GronorNOCIDeterminantPairPreparation
export GronorNOCIDeterminantPairOneBodyElement
export GronorNOCIDeterminantPairMatrixElement
export GronorNOCIDirectTwoElectronWorkspace
export algorithm_name
export backend_name
export estimate_array_bytes
export combine_memory_estimates
export check_memory_limits
export enforce_memory_limits
export record_phase!
export set_counter!
export increment_counter!
export cofactor_workspace_memory
export contracted_complementary_cofactors!
export contracted_complementary_cofactors
export active_response_count
export active_response_index
export ahpc_state_pair_memory
export prepare_ahpc_state_pair
export active_spin_occupied_counts
export raw_noci_matrices_memory
export noci_normalization_memory
export normalize_noci_matrices
export generalized_noci_solver_memory
export solve_generalized_noci
export lowdin_orthogonalization
export gallup_norbeck_weights
export apply_diagonal_shifts
export pairwise_lowdin_couplings
export full_multistate_lowdin_couplings
export build_canonical_mebf_spin_strings
export spin_string_indices
export spin_pair_incidence_memory
export build_spin_pair_incidence
export spin_pair_intermediate_memory
export build_spin_pair_intermediates
export sparse_spin_pair_contraction_memory
export contract_sparse_spin_pair_weights
export assemble_ahpc_spin_string_matrices
export gronor_noci_pair_preparation_memory
export gronor_noci_full_occupied_cofactors
export prepare_gronor_noci_determinant_pair
export contract_gronor_noci_one_body
export gronor_noci_two_electron_memory
export contract_gronor_noci_two_electron!
export contract_gronor_noci_two_electron
export contract_gronor_noci_determinant_pair!
export contract_gronor_noci_determinant_pair
export gronor_noci_matrix_memory
export assemble_gronor_noci_matrices
export precontraction_memory
export precontract_active_hamiltonian
export evaluate
export run_noci
export apply_caspt2_override

end
