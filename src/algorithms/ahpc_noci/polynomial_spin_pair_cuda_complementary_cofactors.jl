
"""
    _polynomial_power_terminal_first_complementary_cofactor_gpu!(...)

Evaluate determinant, retained direct powers, and the first complementary
cofactor from one shared direct-power polynomial backbone.

For occupied rank `q>=2`, the kernel forms `A^0 ... A^(q-2)`, obtains
`e_0 ... e_(q-2)` from the Newton trace identities, constructs
`N_(q-2)`, and applies the exact terminal recurrences

`e_(q-1)=tr(A*N_(q-2))/(q-1)`,
`adj(A)=e_(q-1)I-A*N_(q-2)`, and
`det(A)=tr(A*adj(A))/q`.

The q=1 case is evaluated analytically. All divisions are by fixed positive
integer orders; there is no determinant division, inverse, numerical-rank
classification, pivot test, SVD, or singular/regular branch. The retained
matrix powers and elementary coefficients are the same inputs consumed by the
rank-uniform symmetry-regrouped second complementary-cofactor kernel.
"""
function _polynomial_power_terminal_first_complementary_cofactor_gpu!(
    elementary_symmetric_coefficients,
    matrix_powers,
    newton_cofactor_matrix,
    first_cofactor_output,
    cofactor_scalar_and_contracted_second_output,
    scaled_effective_active_overlap,
    n_active_occupied_device::Int32,
    batch_count::Int32,
)
    union_id =
        (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) +
        Int(CUDA.threadIdx().x)
    union_id <= Int(batch_count) || return nothing

    n_active_occupied = Int(n_active_occupied_device)
    n_active_responses =
        n_active_occupied * n_active_occupied
    T = eltype(matrix_powers)

    @inbounds for order_index in 1:(n_active_occupied + 1)
        elementary_symmetric_coefficients[
            order_index,
            union_id,
        ] = zero(T)
    end
    @inbounds elementary_symmetric_coefficients[1, union_id] = one(T)

    if n_active_occupied == 1
        overlap_value =
            @inbounds scaled_effective_active_overlap[1, union_id]
        @inbounds elementary_symmetric_coefficients[2, union_id] =
            overlap_value
        @inbounds matrix_powers[1, union_id] = one(T)
        @inbounds first_cofactor_output[1, union_id] = one(T)
        @inbounds cofactor_scalar_and_contracted_second_output[
            1,
            union_id,
        ] = overlap_value
        @inbounds cofactor_scalar_and_contracted_second_output[
            2,
            union_id,
        ] = zero(T)
        return nothing
    end

    # A^0 = I.
    @inbounds for column in 1:n_active_occupied,
                  row in 1:n_active_occupied
        active_response =
            row + (column - 1) * n_active_occupied
        matrix_powers[active_response, union_id] =
            row == column ? one(T) : zero(T)
    end

    # Build A^1 ... A^(q-2).  Until the terminal Newton matrix is needed,
    # the first q-2 entries of the q-by-q Newton scratch hold power traces.
    if n_active_occupied > 2
        @inbounds for exponent in 1:(n_active_occupied - 2)
            current_power_offset =
                exponent * n_active_responses
            if exponent == 1
                for active_response in 1:n_active_responses
                    matrix_powers[
                        current_power_offset + active_response,
                        union_id,
                    ] =
                        scaled_effective_active_overlap[
                            active_response,
                            union_id,
                        ]
                end
            else
                previous_power_offset =
                    (exponent - 1) * n_active_responses
                for column in 1:n_active_occupied,
                    row in 1:n_active_occupied
                    value = zero(T)
                    for inner in 1:n_active_occupied
                        value +=
                            matrix_powers[
                                previous_power_offset +
                                row +
                                (inner - 1) * n_active_occupied,
                                union_id,
                            ] *
                            scaled_effective_active_overlap[
                                inner +
                                (column - 1) * n_active_occupied,
                                union_id,
                            ]
                    end
                    matrix_powers[
                        current_power_offset +
                        row +
                        (column - 1) * n_active_occupied,
                        union_id,
                    ] = value
                end
            end

            trace_value = zero(T)
            for diagonal in 1:n_active_occupied
                trace_value +=
                    matrix_powers[
                        current_power_offset +
                        diagonal +
                        (diagonal - 1) * n_active_occupied,
                        union_id,
                    ]
            end
            newton_cofactor_matrix[
                exponent,
                union_id,
            ] = trace_value
        end

        # Newton scalar identities:
        # e_m = (1/m) sum_j (-1)^(j-1) e_(m-j) tr(A^j).
        for order in 1:(n_active_occupied - 2)
            value = zero(T)
            for power_order in 1:order
                term =
                    elementary_symmetric_coefficients[
                        order - power_order + 1,
                        union_id,
                    ] *
                    newton_cofactor_matrix[
                        power_order,
                        union_id,
                    ]
                value +=
                    isodd(power_order) ? term : -term
            end
            elementary_symmetric_coefficients[
                order + 1,
                union_id,
            ] = value / order
        end
    end

    # N_(q-2) overwrites the temporary trace storage.
    @inbounds for column in 1:n_active_occupied,
                  row in 1:n_active_occupied
        active_response =
            row + (column - 1) * n_active_occupied
        value = zero(T)
        for power_order in 0:(n_active_occupied - 2)
            coefficient =
                elementary_symmetric_coefficients[
                    n_active_occupied - 1 - power_order,
                    union_id,
                ]
            isodd(power_order) &&
                (coefficient = -coefficient)
            value +=
                coefficient *
                matrix_powers[
                    power_order * n_active_responses +
                    active_response,
                    union_id,
                ]
        end
        newton_cofactor_matrix[
            active_response,
            union_id,
        ] = value
    end

    # e_(q-1) = tr(A*N_(q-2))/(q-1).
    terminal_trace = zero(T)
    @inbounds for diagonal in 1:n_active_occupied
        diagonal_value = zero(T)
        for inner in 1:n_active_occupied
            diagonal_value +=
                scaled_effective_active_overlap[
                    diagonal +
                    (inner - 1) * n_active_occupied,
                    union_id,
                ] *
                newton_cofactor_matrix[
                    inner +
                    (diagonal - 1) * n_active_occupied,
                    union_id,
                ]
        end
        terminal_trace += diagonal_value
    end
    elementary_q_minus_1 =
        terminal_trace / (n_active_occupied - 1)
    @inbounds elementary_symmetric_coefficients[
        n_active_occupied,
        union_id,
    ] = elementary_q_minus_1

    # first_cofactor[row,column] = adj(A)[column,row].
    @inbounds for column in 1:n_active_occupied,
                  row in 1:n_active_occupied
        terminal_transposed = zero(T)
        for inner in 1:n_active_occupied
            terminal_transposed +=
                scaled_effective_active_overlap[
                    column +
                    (inner - 1) * n_active_occupied,
                    union_id,
                ] *
                newton_cofactor_matrix[
                    inner +
                    (row - 1) * n_active_occupied,
                    union_id,
                ]
        end
        first_cofactor_output[
            row + (column - 1) * n_active_occupied,
            union_id,
        ] =
            (row == column ?
                elementary_q_minus_1 :
                zero(T)) -
            terminal_transposed
    end

    determinant_trace = zero(T)
    @inbounds for column in 1:n_active_occupied,
                  row in 1:n_active_occupied
        active_response =
            row + (column - 1) * n_active_occupied
        determinant_trace +=
            scaled_effective_active_overlap[
                active_response,
                union_id,
            ] *
            first_cofactor_output[
                active_response,
                union_id,
            ]
    end
    @inbounds cofactor_scalar_and_contracted_second_output[
        1,
        union_id,
    ] =
        determinant_trace / n_active_occupied
    @inbounds cofactor_scalar_and_contracted_second_output[
        2,
        union_id,
    ] = zero(T)
    return nothing
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
@inline function _polynomial_second_complementary_cofactor_regrouped_matrix_gpu!(
    regrouped_matrix,
    elementary_symmetric_coefficients,
    matrix_powers,
    n_active_responses::Int,
    union_id::Int,
    grouped_power::Int,
    n_active_occupied::Int,
    orbital_pair_lane::Int,
    orbital_pair_mask::UInt32,
)
    diagonal_elementary_order =
        n_active_occupied - 2 - 2 * grouped_power
    diagonal_coefficient =
        @inbounds elementary_symmetric_coefficients[
            diagonal_elementary_order + 1,
            union_id,
        ]
    grouped_power_offset = grouped_power * n_active_responses

    active_response = orbital_pair_lane
    while active_response <= n_active_responses
        @inbounds regrouped_matrix[active_response, union_id] =
            diagonal_coefficient *
            matrix_powers[
                grouped_power_offset + active_response,
                union_id,
            ]
        active_response +=
            AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_UNION
    end

    highest_right_power =
        n_active_occupied - 2 - grouped_power
    @inbounds for right_power in
        (grouped_power + 1):highest_right_power
        elementary_order =
            n_active_occupied - 2 - grouped_power - right_power
        coefficient =
            elementary_symmetric_coefficients[
                elementary_order + 1,
                union_id,
            ]
        isodd(grouped_power + right_power) &&
            (coefficient = -coefficient)
        paired_coefficient = coefficient + coefficient
        right_power_offset = right_power * n_active_responses

        active_response = orbital_pair_lane
        while active_response <= n_active_responses
            regrouped_matrix[active_response, union_id] +=
                paired_coefficient *
                matrix_powers[
                    right_power_offset + active_response,
                    union_id,
                ]
            active_response +=
                AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_UNION
        end
    end
    CUDA.sync_warp(orbital_pair_mask)
    return nothing
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
function _polynomial_second_complementary_cofactor_endpoint_gpu!(
    elementary_symmetric_coefficients,
    matrix_powers,
    regrouped_matrix,
    cofactor_scalar_and_contracted_second_output,
    hessian_row_pair_indices,
    hessian_column_pair_indices,
    state_pair_hessian_response_device,
    n_active_occupied_device::Int32,
    batch_count::Int32,
)
    occupied_orbital_pairs =
        CUDA.CuStaticSharedArray(
            Int16,
            2 * AHPC_MAX_OCCUPIED_ORBITAL_PAIRS,
        )

    thread_in_block = Int(CUDA.threadIdx().x)
    n_threads_in_block = Int(CUDA.blockDim().x)
    n_active_occupied = Int(n_active_occupied_device)
    n_occupied_orbital_pairs =
        (n_active_occupied * (n_active_occupied - 1)) ÷ 2

    pair_index = thread_in_block
    while pair_index <= n_occupied_orbital_pairs
        first_occupied, second_occupied =
            _occupied_orbital_pair_from_index(
                pair_index,
                n_active_occupied,
            )
        @inbounds occupied_orbital_pairs[pair_index] =
            Int16(first_occupied)
        @inbounds occupied_orbital_pairs[
            AHPC_MAX_OCCUPIED_ORBITAL_PAIRS + pair_index
        ] = Int16(second_occupied)
        pair_index += n_threads_in_block
    end
    CUDA.sync_threads()

    global_thread =
        (Int(CUDA.blockIdx().x) - 1) * n_threads_in_block +
        thread_in_block
    global_warp = (global_thread - 1) ÷ 32
    warp_lane_zero = mod(global_thread - 1, 32)

    n_unions_per_warp =
        32 ÷ AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_UNION
    union_slot = mod(warp_lane_zero, n_unions_per_warp)
    orbital_pair_lane =
        warp_lane_zero ÷ n_unions_per_warp + 1
    union_id =
        global_warp * n_unions_per_warp + union_slot + 1
    union_id <= Int(batch_count) || return nothing

    orbital_pair_mask = UInt32(0)
    local_lane = 0
    while local_lane <
          AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_UNION
        bit_index =
            union_slot + local_lane * n_unions_per_warp
        orbital_pair_mask |= UInt32(1) << bit_index
        local_lane += 1
    end

    n_active_responses = n_active_occupied * n_active_occupied
    _polynomial_second_complementary_cofactor_regrouped_matrix_gpu!(
        regrouped_matrix,
        elementary_symmetric_coefficients,
        matrix_powers,
        n_active_responses,
        union_id,
        0,
        n_active_occupied,
        orbital_pair_lane,
        orbital_pair_mask,
    )

    endpoint = zero(eltype(cofactor_scalar_and_contracted_second_output))
    @inbounds for column_pair_index in 1:n_occupied_orbital_pairs
        first_column = Int(occupied_orbital_pairs[column_pair_index])
        second_column = Int(occupied_orbital_pairs[
            AHPC_MAX_OCCUPIED_ORBITAL_PAIRS + column_pair_index,
        ])
        state_pair_column_pair = Int(hessian_column_pair_indices[
            column_pair_index,
            union_id,
        ])

        n_overlapping_row_pairs = 2 * n_active_occupied - 3
        overlap_start = 1
        while overlap_start <= n_overlapping_row_pairs
            overlap_index = overlap_start + orbital_pair_lane - 1
            pair_term = zero(eltype(cofactor_scalar_and_contracted_second_output))
            if overlap_index <= n_overlapping_row_pairs
                first_row, second_row, row_pair_index =
                    _overlapping_occupied_orbital_pair_from_index(
                        overlap_index,
                        first_column,
                        second_column,
                        n_active_occupied,
                    )
                pair_term = _second_complementary_cofactor_identity_pair_term(
                    regrouped_matrix,
                    hessian_row_pair_indices,
                    state_pair_hessian_response_device,
                    union_id,
                    0,
                    first_column,
                    second_column,
                    first_row,
                    second_row,
                    row_pair_index,
                    state_pair_column_pair,
                    n_active_occupied,
                )
            end

            for source_lane in 1:AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_UNION
                source_warp_lane =
                    (source_lane - 1) * n_unions_per_warp + union_slot + 1
                gathered_pair_term = CUDA.shfl_sync(
                    orbital_pair_mask,
                    pair_term,
                    source_warp_lane,
                )
                orbital_pair_lane == 1 && (endpoint += gathered_pair_term)
            end
            overlap_start += AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_UNION
        end
    end

    if orbital_pair_lane == 1
        @inbounds cofactor_scalar_and_contracted_second_output[
            2,
            union_id,
        ] = endpoint
    end
    return nothing
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
function _polynomial_second_complementary_cofactor_blocked_shared_gpu!(
    elementary_symmetric_coefficients,
    matrix_powers,
    cofactor_scalar_and_contracted_second_output,
    hessian_row_pair_indices,
    hessian_column_pair_indices,
    state_pair_hessian_response_device,
    n_active_occupied_device::Int32,
    batch_count::Int32,
    stage_groups_device::Int32,
)
    thread_id = Int(CUDA.threadIdx().x)
    n_threads = Int(CUDA.blockDim().x)
    union_id = Int(CUDA.blockIdx().x)
    union_id <= Int(batch_count) || return nothing

    n_active_occupied = Int(n_active_occupied_device)
    highest_grouped_power = (n_active_occupied - 2) ÷ 2
    highest_grouped_power >= 1 || return nothing
    stage_groups = Int(stage_groups_device)
    n_active_responses = n_active_occupied * n_active_occupied
    n_occupied_orbital_pairs =
        (n_active_occupied * (n_active_occupied - 1)) ÷ 2
    T = eltype(matrix_powers)

    occupied_orbital_pairs = CUDA.CuStaticSharedArray(
        Int16,
        2 * AHPC_MAX_OCCUPIED_ORBITAL_PAIRS,
    )
    state_pair_hessian_pair_indices = CUDA.CuStaticSharedArray(
        Int32,
        2 * AHPC_MAX_OCCUPIED_ORBITAL_PAIRS,
    )

    pair_index = thread_id
    while pair_index <= n_occupied_orbital_pairs
        first_occupied, second_occupied = _occupied_orbital_pair_from_index(
            pair_index,
            n_active_occupied,
        )
        @inbounds occupied_orbital_pairs[pair_index] = Int16(first_occupied)
        @inbounds occupied_orbital_pairs[
            AHPC_MAX_OCCUPIED_ORBITAL_PAIRS + pair_index,
        ] = Int16(second_occupied)
        @inbounds state_pair_hessian_pair_indices[pair_index] =
            Int32(hessian_row_pair_indices[pair_index, union_id])
        @inbounds state_pair_hessian_pair_indices[
            AHPC_MAX_OCCUPIED_ORBITAL_PAIRS + pair_index,
        ] = Int32(hessian_column_pair_indices[pair_index, union_id])
        pair_index += n_threads
    end

    matrix_groups = max(stage_groups, 1)
    shared_entry_count =
        2 * matrix_groups * n_active_responses + n_threads
    shared = CUDA.CuDynamicSharedArray(T, shared_entry_count)
    reduction_offset = 2 * matrix_groups * n_active_responses
    CUDA.sync_threads()

    dense_total = zero(T)
    block_start = 1
    while block_start <= highest_grouped_power
        active_groups = min(
            stage_groups,
            highest_grouped_power - block_start + 1,
        )

        group_local = 0
        while group_local < active_groups
            grouped_power = block_start + group_local
            left_shared_offset = 2 * group_local * n_active_responses
            right_shared_offset = left_shared_offset + n_active_responses
            left_global_offset = grouped_power * n_active_responses

            active_response = thread_id
            while active_response <= n_active_responses
                left_value = @inbounds matrix_powers[
                    left_global_offset + active_response,
                    union_id,
                ]
                @inbounds shared[left_shared_offset + active_response] =
                    left_value

                diagonal_order =
                    n_active_occupied - 2 - 2 * grouped_power
                coefficient = @inbounds elementary_symmetric_coefficients[
                    diagonal_order + 1,
                    union_id,
                ]
                right_value = coefficient * left_value
                highest_right_power =
                    n_active_occupied - 2 - grouped_power
                right_power = grouped_power + 1
                while right_power <= highest_right_power
                    elementary_order =
                        n_active_occupied - 2 - grouped_power - right_power
                    right_coefficient =
                        @inbounds elementary_symmetric_coefficients[
                            elementary_order + 1,
                            union_id,
                        ]
                    isodd(grouped_power + right_power) &&
                        (right_coefficient = -right_coefficient)
                    right_value +=
                        (right_coefficient + right_coefficient) *
                        matrix_powers[
                            right_power * n_active_responses + active_response,
                            union_id,
                        ]
                    right_power += 1
                end
                @inbounds shared[right_shared_offset + active_response] =
                    right_value
                active_response += n_threads
            end
            group_local += 1
        end
        CUDA.sync_threads()

        partial = zero(T)
        dense_index = thread_id
        dense_count =
            n_occupied_orbital_pairs * n_occupied_orbital_pairs
        while dense_index <= dense_count
            row_pair_index =
                mod(dense_index - 1, n_occupied_orbital_pairs) + 1
            column_pair_index =
                div(dense_index - 1, n_occupied_orbital_pairs) + 1

            first_row = Int(occupied_orbital_pairs[row_pair_index])
            second_row = Int(occupied_orbital_pairs[
                AHPC_MAX_OCCUPIED_ORBITAL_PAIRS + row_pair_index,
            ])
            first_column = Int(occupied_orbital_pairs[column_pair_index])
            second_column = Int(occupied_orbital_pairs[
                AHPC_MAX_OCCUPIED_ORBITAL_PAIRS + column_pair_index,
            ])
            state_pair_row_pair =
                Int(state_pair_hessian_pair_indices[row_pair_index])
            state_pair_column_pair = Int(state_pair_hessian_pair_indices[
                AHPC_MAX_OCCUPIED_ORBITAL_PAIRS + column_pair_index,
            ])
            response = @inbounds state_pair_hessian_response_device[
                state_pair_row_pair,
                state_pair_column_pair,
            ]

            grouped_sum = zero(T)
            group_local = 0
            while group_local < active_groups
                left_base = 2 * group_local * n_active_responses
                right_base = left_base + n_active_responses
                wedge_bilinear =
                    @inbounds shared[
                        left_base +
                        first_column +
                        (first_row - 1) * n_active_occupied,
                    ] * shared[
                        right_base +
                        second_column +
                        (second_row - 1) * n_active_occupied,
                    ]
                wedge_bilinear -=
                    @inbounds shared[
                        left_base +
                        second_column +
                        (first_row - 1) * n_active_occupied,
                    ] * shared[
                        right_base +
                        first_column +
                        (second_row - 1) * n_active_occupied,
                    ]
                wedge_bilinear -=
                    @inbounds shared[
                        left_base +
                        first_column +
                        (second_row - 1) * n_active_occupied,
                    ] * shared[
                        right_base +
                        second_column +
                        (first_row - 1) * n_active_occupied,
                    ]
                wedge_bilinear +=
                    @inbounds shared[
                        left_base +
                        second_column +
                        (second_row - 1) * n_active_occupied,
                    ] * shared[
                        right_base +
                        first_column +
                        (first_row - 1) * n_active_occupied,
                    ]
                grouped_sum += wedge_bilinear
                group_local += 1
            end
            partial += response * grouped_sum
            dense_index += n_threads
        end

        @inbounds shared[reduction_offset + thread_id] = partial
        CUDA.sync_threads()
        stride = n_threads ÷ 2
        while stride >= 1
            if thread_id <= stride
                @inbounds shared[reduction_offset + thread_id] +=
                    shared[reduction_offset + thread_id + stride]
            end
            CUDA.sync_threads()
            stride ÷= 2
        end
        thread_id == 1 &&
            (dense_total += @inbounds shared[reduction_offset + 1])
        CUDA.sync_threads()
        block_start += stage_groups
    end

    if thread_id == 1
        @inbounds cofactor_scalar_and_contracted_second_output[
            2,
            union_id,
        ] += dense_total
    end
    return nothing
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
function _launch_polynomial_complementary_cofactors_gpu!(
    buffers::Tuple,
    n_active_occupied_device::Int32,
    width::Int32,
)
    n_active_occupied = Int(n_active_occupied_device)
    batch_width = Int(width)

    1 <= n_active_occupied <= AHPC_MAX_OCCUPIED_RANK ||
        throw(ArgumentError(
            "polynomial complementary cofactors require occupied rank 1:$(AHPC_MAX_OCCUPIED_RANK)",
        ))
    batch_width >= 1 ||
        throw(ArgumentError(
            "polynomial complementary-cofactor batch width must be positive",
        ))

    recurrence_buffers =
        ntuple(
            index -> _union_contiguous_polynomial_recurrence_buffer(
                buffers[index],
            ),
            Val(3),
        )
    cofactor_buffers = (
        recurrence_buffers...,
        buffers[4],
        buffers[5],
        buffers[6],
        buffers[7],
        buffers[8],
        buffers[9],
    )

    first_cofactor_kernel =
        CUDA.@cuda threads=AHPC_POLYNOMIAL_GPU_THREADS blocks=cld(
            batch_width,
            AHPC_POLYNOMIAL_GPU_THREADS,
        ) _polynomial_power_terminal_first_complementary_cofactor_gpu!(
            cofactor_buffers[1],
            cofactor_buffers[2],
            cofactor_buffers[3],
            cofactor_buffers[4],
            cofactor_buffers[5],
            cofactor_buffers[6],
            n_active_occupied_device,
            width,
        )

    n_active_occupied == 1 && return first_cofactor_kernel

    n_warps_per_block =
        AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_BLOCK ÷ 32
    n_unions_per_warp =
        32 ÷ AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_UNION
    n_unions_per_block =
        n_warps_per_block * n_unions_per_warp
    endpoint_blocks = cld(batch_width, n_unions_per_block)

    endpoint_kernel = CUDA.@cuda threads=AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_BLOCK blocks=endpoint_blocks _polynomial_second_complementary_cofactor_endpoint_gpu!(
        cofactor_buffers[1],
        cofactor_buffers[2],
        cofactor_buffers[3],
        cofactor_buffers[5],
        cofactor_buffers[7],
        cofactor_buffers[8],
        cofactor_buffers[9],
        n_active_occupied_device,
        width,
    )

    highest_grouped_power = (n_active_occupied - 2) ÷ 2
    highest_grouped_power == 0 && return endpoint_kernel

    T = eltype(cofactor_buffers[2])
    stage_groups =
        _second_complementary_cofactor_blocked_stage_groups(
            T,
            n_active_occupied,
        )
    shared_bytes =
        _second_complementary_cofactor_blocked_shared_bytes(
            T,
            n_active_occupied,
        )
    return CUDA.@cuda threads=AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_BLOCK blocks=batch_width shmem=shared_bytes _polynomial_second_complementary_cofactor_blocked_shared_gpu!(
        cofactor_buffers[1],
        cofactor_buffers[2],
        cofactor_buffers[5],
        cofactor_buffers[7],
        cofactor_buffers[8],
        cofactor_buffers[9],
        n_active_occupied_device,
        width,
        Int32(stage_groups),
    )
end
