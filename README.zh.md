# NOCI.jl

[English](README.md)

NOCI.jl 用 Julia 实现行列式对 NOCI（DP-NOCI）与活性 Hamiltonian 预收缩 NOCI（AHPC-NOCI）。
程序从行列式、轨道和积分数据计算原始及重叠归一化的 $S/H_1/H_2/H$ 矩阵，并求解对称广义 NOCI 本征值问题。
本日历版本提供 CPU 执行，版本权威是 `Project.toml`，CLI、科学输出和 JSON 一致报告完整版本。

## 安装和准备

先读 [安装指南](Install/README.zh.md)，再使用 [schema-4 标准流程](examples/ahpc_noci_standard/TUTORIAL.zh.md)。
默认安装包括独立编号的 CAS/积分后端 OpenMolcas4NOCI 26.06.2，适配由 `OpenMolcasCompatibility.toml` 声明和检查。
已有兼容积分数据时可仅安装 CPU。

```bash
bash Install/install.sh --plan
# 下一行会安装并执行小型安装检查。
bash Install/install.sh --execute
```

推荐 C1 流程共享原始单体 CAS 任务，逐实例旋转轨道，独立构造 CommonOrb 并导入 RICD/CD 因子供 AHPC-NOCI 使用。
单体所选根的 CASPT2 默认开启，可显式关闭。
PySCF DFCD 和混搭后端仅有 [预留契约](examples/ahpc_noci_standard/BACKEND_CONTRACT.zh.md)，本版没有可执行的完整适配器。
科学运行不安装、下载或求解包。

## 算法和数据

`DPNOCI` 对物理四指标 ERI 求值，可直接读取 packed `.two`；显式指定的嵌入 SEWARD 来源先在内存预检后物化四指标积分。
DP-NOCI 拒绝直接输入因子包。
`AHPCNOCI` 结合活性 Hamiltonian 预收缩、规范自旋字符串对复用、稀疏收缩和 occupied-spin 互补余子式。
其单一生产路径支持 packed 四指标或分子 CD、度量白化 RICD 的单位度量 Gram 因子。
支持查询和 polynomial/带证书 LU 余子式仍是类型化算法选项，预收缩是 AHPC 的基础，不是第三种方法。
大规模常规四指标计算需要主机内存预检；推荐因子流程不形成完整四指标矩阵。

可移植 SEWARD CHH5 经过验证后，直接变换到实际公共轨道基并处理冻结核心，HDF5 导入器不调用共享库。
原嵌入 RunFile/SEWARD Julia-MOTRA 仍是专家输入路线，使用共享库时必须符合声明的后端适配。
新输入使用 `integrals.backend: factorized`，旧 `cholesky` 拼写和版本 1/2 因子包仍可读。
RICD 原子辅助基阈值与分子 CD 阈值在来源记录中保持不同含义。
导入器支持 CHH5 1.0 和显式来源 1.1，不推断旧数据未知来源。
标准算例生成 `factors.jld2`，独立转换工具保留各自说明的前缀命名。

## 命令

```bash
noci --version
noci run INPUT.yaml OUTPUT_DIRECTORY OUTPUT_ROOT
noci override SNAPSHOT.jld2 OVERRIDE.yaml
noci-prepare init NEW_CASE_DIRECTORY
noci-prepare check MANIFEST.yaml
noci-prepare plan MANIFEST.yaml
```

生产输出保留未修正与相关能修正的矩阵/能量、标量态耦合、来源、快照和可选续算断点。
真实计算命令和输入假设见 [准备指南](docs/tutorials/AHPC_NOCI_Preparation_and_Run.zh.md) 与 [标准教程](examples/ahpc_noci_standard/TUTORIAL.zh.md)。
产品不分发开发测试、性能输出、机器配置和发布管理材料。

## 科学引用

If you use NOCI.jl — through either the AHPC-NOCI or the DP-NOCI module — you must cite the paper that introduced the reduced common molecular orbital basis (`COMMONORB`), the foundation of the NOCI-F method on which both algorithms rest:

> R. K. Kathir, C. de Graaf, R. Broer, and R. W. A. Havenith, "Reduced Common Molecular Orbital Basis for Nonorthogonal Configuration Interaction", *J. Chem. Theory Comput.* **16**, 2941–2951 (2020), doi:[10.1021/acs.jctc.9b01144](https://doi.org/10.1021/acs.jctc.9b01144)

## 来源和致谢

The determinant-pair method (DP-NOCI) originates from the GronOR program:

> T. P. Straatsma, R. Broer, A. Sanchez-Mansilla, C. Sousa, and C. de Graaf, "GronOR: Scalable and Accelerated Nonorthogonal Configuration Interaction for Molecular Fragment Wave Functions", *J. Chem. Theory Comput.* **18**, 3549–3565 (2022), doi:[10.1021/acs.jctc.2c00266](https://doi.org/10.1021/acs.jctc.2c00266)

GronOR ([gitlab.com/gronor/gronor](https://gitlab.com/gronor/gronor)) is copyright of the University of Groningen and is distributed under the Apache License 2.0, the same license as this package. NOCI.jl rewrites the method in Julia with extensive single-node performance optimizations, keeping most functionality and settings consistent with GronOR; the provenance is also declared in the header of every `src/algorithms/dp_noci/` file.

The fragment correlation-energy (CASPT2) shift with Gallup-Norbeck weighting, implemented in `src/core/correlation_shifts.jl`, likewise originates from GronOR (routine `gronor_correlation_energy`, Kathir, Havenith, and de Graaf, 2019, published in the paper above) and is rewritten in Julia with both the GronOR Fortran and the original paper weight conventions.


