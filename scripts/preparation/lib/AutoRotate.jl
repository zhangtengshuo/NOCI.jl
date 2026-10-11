# AutoRotate: 自动计算优化单体 → 晶体多分子结构位置的旋转平移矩阵 + 合并优化 cluster xyz
#
# 替代标准 grotate 流程里"手动指定对应原子"的前置几何步骤:
#   输入: original_ABCX.xyz (晶体 cluster) + 任意个优化单体 xyz
#   输出: matrix_A.txt, matrix_B.txt, ... (R 3x3 + t 3), MolAB[CDE...].xyz + RMSD 报告
#
# 算法 (排除 H, 只用重原子做对应与拟合):
#   1. 鲁棒解析 3 个 xyz (容忍 "!" 注释行 / 能量行 / merged title)
#   2. original cluster 按输入单体顺序和原子数依次切块
#   3. 每个单体: 优化重原子 ↔ original 该单体同元素重原子, 用距离指纹贪心匹配建立对应
#   4. Kabsch SVD (全部重原子最小二乘) 求 R, t:  x_orig = R·x_opt + t
#   5. 合并 xyz: 各优化单体全原子经对应 R/t 变换后按输入顺序合并
#   6. 报告重原子 RMSD；若超过阈值则报错, 视为结构/顺序不一致
#
# Kabsch 数学: 给定对应点集 P(opt), Q(orig), 最小化 sum_i ||Q_i - (R P_i + t)||^2
#   mu_P = mean(P), mu_Q = mean(Q);  H = (P-mu_P)^T (Q-mu_Q);  H = U S V^T
#   R = V diag(1,1, sign(det(V U^T))) U^T ;  t = mu_Q - R mu_P

using LinearAlgebra
using Printf

struct Atom
    element::String
    coord::Vector{Float64}
end

isheavy(a::Atom) = a.element != "H" && a.element != "h"

# --- xyz I/O (鲁棒: 容忍行1带注释、行2 为任意 title/能量) ---

function read_xyz(path::AbstractString)::Vector{Atom}
    isfile(path) || error("xyz not found: $path")
    lines = [l for l in readlines(path) if !isempty(strip(l))]
    natoms = parse(Int, split(lines[1])[1])
    natoms + 2 <= length(lines) ||
        error("$path: header says $natoms atoms but file has only $(length(lines)) lines")
    atoms = Vector{Atom}(undef, natoms)
    for k in 1:natoms
        s = split(lines[2 + k])
        atoms[k] = Atom(s[1], parse.(Float64, s[2:4]))
    end
    return atoms
end

function write_xyz(path::AbstractString, atoms::Vector{Atom}, title::AbstractString)
    open(path, "w") do io
        println(io, length(atoms))
        println(io, title)
        for a in atoms
            @printf(io, "%-2s %18.10f %18.10f %18.10f\n", a.element, a.coord...)
        end
    end
end

function write_matrix(path::AbstractString, R::Matrix{Float64}, t::Vector{Float64}, label::AbstractString)
    open(path, "w") do io
        println(io, "# rotation-translation for monomer $label")
        println(io, "# convention:  x_orig = R * x_opt + t")
        println(io, "# R (3x3, row-major):")
        for i in 1:3
            @printf(io, "  %22.14e %22.14e %22.14e\n", R[i, 1], R[i, 2], R[i, 3])
        end
        println(io, "# t (3):")
        @printf(io, "  %22.14e %22.14e %22.14e\n", t[1], t[2], t[3])
    end
end

# --- cluster 划分 ---

function elem_counts(atoms::Vector{Atom})::Dict{String,Int}
    d = Dict{String,Int}()
    for a in atoms
        d[a.element] = get(d, a.element, 0) + 1
    end
    return d
end

function monomer_label(i::Integer)
    i > 0 || throw(ArgumentError("monomer index must be positive"))
    chars = Char[]
    n = Int(i)
    while n > 0
        n -= 1
        pushfirst!(chars, Char(Int('A') + (n % 26)))
        n ÷= 26
    end
    return String(chars)
end

cluster_xyz_name(n::Integer) = "Mol" * join(monomer_label(i) for i in 1:n) * ".xyz"

# original cluster 按输入单体顺序切分；不自动重排分子块。
# 每一块的元素计数必须与对应优化单体一致，否则说明输入顺序或结构不符合预期。
function split_cluster(cluster::Vector{Atom}, monomers::Vector{Vector{Atom}})
    expected = sum(length, monomers)
    length(cluster) == expected ||
        error("original cluster has $(length(cluster)) atoms, expected sum of monomers = $expected")
    blocks = Vector{Vector{Atom}}(undef, length(monomers))
    start = 1
    for i in eachindex(monomers)
        n = length(monomers[i])
        block = cluster[start:start+n-1]
        elem_counts(block) == elem_counts(monomers[i]) ||
            error("original cluster block $(monomer_label(i)) element counts do not match monomer $(i); input monomer order is inconsistent")
        blocks[i] = block
        start += n
    end
    return blocks
end

# --- 重原子距离指纹 + 贪心匹配 ---

# 原子 i 的指纹 = 到其他重原子的距离升序向量 (对整体构象轻微变化稳定)
function fingerprint(coords::Vector{Vector{Float64}}, i::Int)::Vector{Float64}
    n = length(coords)
    d = Vector{Float64}(undef, n - 1)
    k = 0
    for j in 1:n
        j == i && continue
        k += 1
        d[k] = norm(coords[i] .- coords[j])
    end
    return sort!(d)
end

sqdist(a::Vector{Float64}, b::Vector{Float64}) = sum((a .- b) .^ 2)

# 贪心匹配: orig 重原子 j ↔ opt 重原子 perm[j] (指纹平方距离最近, 不重复)
# 对称等价原子指纹相同 → 互换, Kabsch 下无害
function match_heavy(opt_coords::Vector{Vector{Float64}}, opt_elem::Vector{String},
                     orig_coords::Vector{Vector{Float64}}, orig_elem::Vector{String})
    n = length(opt_coords)
    @assert length(orig_coords) == n
    elem_counts([Atom(e, zeros(3)) for e in opt_elem]) == elem_counts([Atom(e, zeros(3)) for e in orig_elem]) ||
        error("heavy atom element counts differ between optimized and original monomer")
    f_opt  = [fingerprint(opt_coords,  i) for i in 1:n]
    f_orig = [fingerprint(orig_coords, j) for j in 1:n]
    perm = zeros(Int, n); used = falses(n)
    for j in 1:n
        best = Inf; bi = 0
        for i in 1:n
            used[i] && continue
            opt_elem[i] == orig_elem[j] || continue
            dd = sqdist(f_orig[j], f_opt[i])
            if dd < best
                best = dd; bi = i
            end
        end
        bi != 0 || error("failed to match heavy atom $(orig_elem[j]) in original monomer")
        perm[j] = bi; used[bi] = true
    end
    return perm
end

# --- Kabsch SVD ---

function kabsch(P::Matrix{Float64}, Q::Matrix{Float64})
    n = size(P, 1)
    mu_P = vec(sum(P; dims = 1)) / n
    mu_Q = vec(sum(Q; dims = 1)) / n
    Pc = P .- mu_P'
    Qc = Q .- mu_Q'
    H = Pc' * Qc                       # 3×3 协方差
    F = svd(H)                         # F.U, F.S, F.V, F.Vt (不要 U,S,Vt = svd(...))
    d = sign(det(F.V * F.U'))
    D = diagm([1.0, 1.0, d])
    R = F.V * D * F.U'
    t = mu_Q - R * mu_P
    return R, t
end

function rmsd_heavy(P::Matrix{Float64}, Q::Matrix{Float64}, R::Matrix{Float64}, t::Vector{Float64})
    n = size(P, 1)
    transformed = (R * P')' .+ t'      # n×3, 每行 R·p_i + t
    return sqrt(sum((transformed .- Q) .^ 2) / n)
end

# --- 单体对齐 ---

# 返回 R, t, 重原子 RMSD。用重原子做对应+拟合, R/t 对全原子(含H)同样适用。
#
# 高对称分子(如 D2h 稠环)的重原子分成少数等价类, 距离指纹无法区分类内原子,
# 贪心匹配可能在等价类内不一致配对(i 配 j, i' 却配到非 j 的对称像)使 Kabsch 偏离。
# 解法: 指纹初值后做 ICP(迭代最近点)——用当前 R/t 变换后按几何最近邻重配对, 迭代
# 到对应关系稳定, 几何最近邻自动给出等价类内的一致配对。
function align_monomer(opt::Vector{Atom}, orig_block::Vector{Atom}; niter::Int = 50,
                       max_rmsd::Float64 = 0.5)
    opt_idx  = findall(isheavy, opt)
    orig_idx = findall(isheavy, orig_block)
    length(opt_idx) == length(orig_idx) ||
        error("heavy atom count: opt=$(length(opt_idx)) vs orig=$(length(orig_idx))")
    opt_c  = [opt[i].coord         for i in opt_idx]
    orig_c = [orig_block[i].coord  for i in orig_idx]
    opt_elem = [opt[i].element for i in opt_idx]
    orig_elem = [orig_block[i].element for i in orig_idx]
    n = length(opt_c)
    perm = match_heavy(opt_c, opt_elem, orig_c, orig_elem) # 初值: 同元素距离指纹贪心匹配
    R = Matrix{Float64}(I, 3, 3); t = zeros(Float64, 3); r = Inf
    for iter in 1:niter
        P = permutedims(reduce(hcat, opt_c[perm])) # n×3, 行 k = opt_c[perm[k]]
        Q = permutedims(reduce(hcat, orig_c))      # n×3, 行 k = orig_c[k]
        R, t = kabsch(P, Q)
        r = rmsd_heavy(P, Q, R, t)
        moved = [R * c + t for c in opt_c]         # 当前 R/t 下的 opt 重原子
        newperm = zeros(Int, n); used = falses(n)
        for j in 1:n                               # 每个 orig 重原子取最近未配 opt
            best = Inf; bi = 0
            for i in 1:n
                used[i] && continue
                opt_elem[i] == orig_elem[j] || continue
                dd = sum(abs2, moved[i] .- orig_c[j])
                if dd < best; best = dd; bi = i; end
            end
            bi != 0 || error("failed to match heavy atom $(orig_elem[j]) during ICP refinement")
            newperm[j] = bi; used[bi] = true
        end
        newperm == perm && break                   # 对应关系稳定, 收敛
        perm = newperm
    end
    r <= max_rmsd ||
        error("heavy-atom RMSD $(round(r; digits=6)) Å exceeds threshold $max_rmsd Å; structure/order is inconsistent")
    return R, t, r
end

transform_atom(a::Atom, R::Matrix{Float64}, t::Vector{Float64}) = Atom(a.element, R * a.coord + t)

# --- 主入口 ---

function auto_rotate(original_path::AbstractString, monomer_paths::AbstractVector{<:AbstractString};
                     outdir::AbstractString = ".", max_rmsd::Float64 = 0.5)
    isempty(monomer_paths) && error("auto_rotate requires at least one optimized monomer xyz")
    cluster = read_xyz(original_path)
    monomers = [read_xyz(path) for path in monomer_paths]
    blocks = split_cluster(cluster, monomers)
    mkpath(outdir)

    rotations = Matrix{Float64}[]
    translations = Vector{Float64}[]
    rmsds = Float64[]
    merged = Atom[]
    title_parts = String[]

    for i in eachindex(monomers)
        label = monomer_label(i)
        R, t, rmsd = align_monomer(monomers[i], blocks[i]; max_rmsd = max_rmsd)
        push!(rotations, R)
        push!(translations, t)
        push!(rmsds, rmsd)
        write_matrix(joinpath(outdir, "matrix_$(label).txt"), R, t,
                     "$label ($(basename(monomer_paths[i])))")
        append!(merged, [transform_atom(a, R, t) for a in monomers[i]])
        push!(title_parts, "$label=$(basename(monomer_paths[i]))")
        @printf("%s (%s): %d heavy atoms, RMSD = %.6f Å\n",
                label, basename(monomer_paths[i]), count(isheavy, monomers[i]), rmsd)
    end

    xyz_name = cluster_xyz_name(length(monomers))
    write_xyz(joinpath(outdir, xyz_name), merged,
              "optimized cluster: $(join(title_parts, " + ")), rotated to crystal frame")

    println("wrote: matrix_*.txt, $xyz_name in $outdir")
    return (R = rotations, t = translations, rmsd = rmsds)
end

function auto_rotate(original_path::AbstractString, monomer_paths::AbstractString...;
                     outdir::AbstractString = ".", max_rmsd::Float64 = 0.5)
    return auto_rotate(original_path, collect(monomer_paths); outdir = outdir, max_rmsd = max_rmsd)
end

function _auto_rotate_usage()
    return "usage: AutoRotate.jl original_ABCX.xyz A.xyz B.xyz ... [--out-dir DIR] [--max-rmsd 0.5]"
end

function _auto_rotate_cli(args::Vector{String} = ARGS)
    outdir = "."
    max_rmsd = 0.5
    files = String[]
    i = 1
    while i <= length(args)
        if args[i] == "--out-dir"
            i + 1 <= length(args) || error("--out-dir needs a value")
            outdir = args[i + 1]
            i += 2
        elseif args[i] == "--max-rmsd"
            i + 1 <= length(args) || error("--max-rmsd needs a value")
            max_rmsd = parse(Float64, args[i + 1])
            i += 2
        else
            push!(files, args[i])
            i += 1
        end
    end
    length(files) >= 2 || error(_auto_rotate_usage())
    auto_rotate(files[1], files[2:end]; outdir = outdir, max_rmsd = max_rmsd)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    _auto_rotate_cli()
end
