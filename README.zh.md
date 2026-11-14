# NOCI.jl

[English](README.md)

**NOCI.jl** 是一个基于分子片段波函数的非正交组态相互作用（nonorthogonal configuration interaction, NOCI）Julia 软件包。 当前方法开发由浙江工业大学化学工程学院的张腾烁和北京师范大学化学学院的崔刚龙教授主导。

程序从片段态构建分子电子基函数（molecular electronic basis functions, MEBFs），计算基函数之间的重叠矩阵元和哈密顿矩阵元，并求解相应的广义本征值问题。

软件包含两种 NOCI 实现：本项目发展的 **AHPC-NOCI**，以及对 GronOR 原始 NOCI-F 方法进行 Julia 重写的 **G-NOCI**。

## 方法

### AHPC-NOCI

AHPC-NOCI 将**活性哈密顿量预收缩（active-Hamiltonian precontraction, AHPC）**、**多项式行列式导数收缩（polynomial determinant-derivative contraction, PDDC）**和**自旋字符串对复用（spin-string-pair reuse）**结合起来。这些技术避免了对大维度重叠矩阵的重复处理，并消除了余子式计算中的 SVD 分支。与传统 NOCI-F 工作流程相比，AHPC-NOCI 的计算性能提升了**四个数量级以上**。

### G-NOCI

G-NOCI 是对 GronOR 原始行列式对 NOCI-F 方法的 Julia 重写。该实现针对**单节点 CPU/GPU 并行计算**进行了高度性能优化。

G-NOCI 保留原方法直接的行列式对计算形式。其计算结构直接、清晰，因此将作为后续新功能开发的首选实现基础，也作为检验 AHPC-NOCI 正确性的参照实现。

## 安装与准备

请先阅读[安装指南](Install/README.zh.md)，再按照 [schema-4 标准流程](examples/ahpc_noci_standard/TUTORIAL.zh.md)准备计算。

默认安装包含 OpenMolcas4NOCI 26.06.2，作为 CAS 和积分计算后端。该后端独立管理版本，并通过 `OpenMolcasCompatibility.toml` 检查兼容性。已有兼容积分数据时，也可以直接用于 NOCI 计算。

```bash
# 查看安装计划
bash Install/install.sh --plan

# 安装 CPU 环境和 OpenMolcas 后端
bash Install/install.sh --execute
```

推荐的 C1 工作流程会复用单体 CAS 任务，对各片段实例进行旋转，构建相应的 CommonOrb 基组，并为 AHPC-NOCI 导入 RICD 或分子 Cholesky 分解（CD）积分因子。选定单体根的 CASPT2 计算默认开启，也可以显式关闭。

## 积分数据

- **G-NOCI** 使用 packed `.two` 文件中的物理四指标电子排斥积分计算行列式对矩阵元。也可以通过显式配置的 SEWARD 数据源生成四指标积分。
- **AHPC-NOCI** 支持 packed 四指标积分，也支持分子 Cholesky 分解（CD）和 resolution-of-identity Cholesky decomposition（RICD）因子化积分。
- OpenMolcas SEWARD 导出的可移植 CHH5 数据经过校验后，会被变换到实际使用的 CommonOrb 基组中，并计入冻结芯层贡献。
- 标准因子化积分输入使用 `integrals.backend: factorized`。

标准算例生成 `factors.jld2`。输入细节见[标准流程教程](examples/ahpc_noci_standard/TUTORIAL.zh.md)和[后端接口约定](examples/ahpc_noci_standard/BACKEND_CONTRACT.zh.md)。

## 计算运行

标准教程介绍了片段态准备、片段放置、CommonOrb 基组构建、积分准备以及 NOCI 矩阵组装。

```bash
# 查看已安装版本
noci --version

# 创建并检查计算案例
noci-prepare init NEW_CASE_DIRECTORY
noci-prepare check MANIFEST.yaml
noci-prepare plan MANIFEST.yaml

# 执行 NOCI 计算
noci run INPUT.yaml OUTPUT_DIRECTORY OUTPUT_ROOT

# 根据覆盖参数重新评估已保存的计算
noci override SNAPSHOT.jld2 OVERRIDE.yaml
```

完整计算流程和输入要求见[准备与运行指南](docs/tutorials/AHPC_NOCI_Preparation_and_Run.zh.md)。

## GPU 计算

默认安装提供 CPU 计算环境。如需安装可选 CUDA 环境，执行：

```bash
bash Install/install.sh --execute --withgpu

# 也可以指定 CUDA runtime
bash Install/install.sh --execute --withgpu --cuda-runtime 12.8
```

标准计算输入默认使用 CPU。若要在 NOCI 矩阵组装阶段（stage 05）使用已安装的 GPU 环境，在 stage-00 manifest 中设置 `runtime.use_gpu: true`。通过 `runtime.gpu_device` 或 `runtime.gpu_devices` 选择 GPU。Julia 线程数必须是所选 GPU 数量的整数倍。CAS、轨道和积分准备阶段使用 CPU。

## 输出

NOCI.jl 保存重叠矩阵、哈密顿矩阵、相关能修正前后的能量、态间耦合、积分来源信息和计算快照。矩阵组装过程可以启用重启检查点，以便续算。

## 引用

**凡使用 NOCI.jl 进行计算并发表相关结果，都必须引用 CommonOrb 论文，无论使用 AHPC-NOCI 还是 G-NOCI。** 两种实现均采用了 NOCI-F 框架所依赖的约化公共分子轨道基组。

> R. K. Kathir, C. de Graaf, R. Broer, and R. W. A. Havenith, “Reduced Common Molecular Orbital Basis for Nonorthogonal Configuration Interaction,” *Journal of Chemical Theory and Computation* **16**, 2941–2951 (2020). [https://doi.org/10.1021/acs.jctc.9b01144](https://doi.org/10.1021/acs.jctc.9b01144)

使用 **G-NOCI** 进行计算时，还应引用介绍其 NOCI-F 方法的 GronOR 原始论文。

> T. P. Straatsma, R. Broer, A. Sanchez-Mansilla, C. Sousa, and C. de Graaf, “GronOR: Scalable and Accelerated Nonorthogonal Configuration Interaction for Molecular Fragment Wave Functions,” *Journal of Chemical Theory and Computation* **18**, 3549–3565 (2022). [https://doi.org/10.1021/acs.jctc.2c00266](https://doi.org/10.1021/acs.jctc.2c00266)

GronOR 原始项目见 [gronor/gronor](https://gitlab.com/gronor/gronor)。
