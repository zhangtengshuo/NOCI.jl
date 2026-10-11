# AHPC-NOCI 通用标准流程详细教程

本教程配套同目录的 `schema_version: 4` YAML 和 00／01／02／03／04／05 文件夹，阅读本目录即可理解输入、执行顺序和产物，不需要追溯旧教程或历史运行目录。
可执行范围是 C1、CPU、OpenMolcas 单体 CAS，以及 OpenMolcas RICD/CD 因子积分的 AHPC-NOCI。
PySCF DFCD 是后续扩展路线，本版提供结构模板与明确的数据契约，不宣称已经实现或验证其数值链路。
在目标环境中使用本目录的独立副本完成实际验收；输入模板和结构检查不能代替真实科学计算。

## 1. 开始前，用户提供什么

先完成需要的单体结构优化、频率检查和目标聚集体结构准备，再开始本流程。
优化和频率不属于 00／01／02／03／04／05，不会在 `run_all.sh` 内隐式执行。
每种分子的每个载体结构需要一份 XYZ，用户还需提供与其结构、基组、AO 中心顺序匹配的初始 RasOrb，以及每个需要计算的态的 CAS 参数。
用户可以为不同态指定同一份 RasOrb，也可以指定各自的文件。
本流程不生成种子、不自动选取或重排活性空间、不判定轨道或根是否具有用户期望的物理性质。
用户对活性空间、初始轨道和态的科学正确性负责；程序只检查输入能否被明确解释、数量与引用是否一致、计算是否正常结束和产物是否可用。

所有 XYZ 坐标使用 Angstrom。
聚集体 XYZ 按 `mebf_table` 的行顺序放置连续的分子块，每块的原子顺序与对应单体 XYZ 和 RasOrb 的 AO 中心顺序一致。
相同元素的原子也不能随意调换，程序不会通过几何猜测去改动轨道所对应的原子身份。

## 2. 分子、结构、位置和态分别如何表示

`molecules.f` 是用户声明的一种分子，例如乙烯；`molecules.g` 可以是丁二烯。
教程推荐使用 f、g、h 等小写短代号，分子注册键允许任意非空 ASCII 字母数字串，也可以直接写 `molecules.ethene`，无需另设 `name` 字段。
分子身份由用户明确声明，程序不通过化学图或 SHA 自动识别。
点群写在 `molecules.f.symmetry: C1`、`molecules.g.symmetry: C1` 等各自的分子定义下，不在 `geometry` 下设置一个全局点群。
这保留了未来异种分子使用不同点群的接口；本版执行范围仍仅 C1，非 C1 定义只能作结构审阅，不进入计算。
逐态 OpenMolcas 的 `symmetry: 1` 是态的不可约表示编号，与分子的点群字段含义不同。
同一种分子可以定义多个载体结构，如 `f/GS0`、`f/GS1`，每个结构有自己的 XYZ 和初始轨道引用。
载体结构名字是引用标识，不能只改文件内容却希望继续复用同名旧结果。

`f_99`、`f_03` 是两个乙烯的位置编号，`g_05` 是一个丁二烯的位置编号。
教程推荐“小写分子代号_两位数字”，但解析器允许 `_` 两侧各为任意非空 ASCII 字母数字串，例如 `ethene_left2`、`9_A7`，数字不限制为 01–99。
标签必须恰有一个 `_`，不接受空格、连字符、额外下划线或路径字符；标签区分大小写，片段标签左侧不用于自动推断分子，实际引用以 `fragments.<label>.molecule` 为准。
片段的规范顺序只由 `mebf_table` 行顺序确定，不能从编号大小或 YAML 映射的排列推断。

`f/GS0/S1` 是旋转前的单体 CAS 任务。
当 f_99 和 f_03 都引用 f/GS0/S1，它们共享同一份算好的原始 RasOrb 和对应 CI 数据；02 再分别旋转到各自的位置，产生 f_99_S1.orb 和 f_03_S1.orb。
03 的 `.vec` 是对应摆放和公共轨道基下的表示，不能把一个位置的 `.vec` 任意当作另一个位置的 `.vec`。
共享原始结果与逐实例摆放同时成立，不需要重复做单体 CAS。

## 3. 选择安装版本和建立工作目录

使用已完成安装的 CPU NOCI 和 OpenMolcas4NOCI；RICD CHH5 需要 兼容配置中明确接受的 OpenMolcas4NOCI v26.06.2。
安装的组件、路径和版本来自安装记录，不在科学运行期间调用包管理器。
CPU 路线不要求 CUDA.jl、GPU、CondaPkg 或 MicroMamba.jl。
OpenMolcas 安装时使用的独立 micromamba 是安装工具，不是运行本流程所需加载的 Julia 包。

指定准备程序所属的确切 NOCI 版本，不使用根级跨版本的 `noci-prepare`：

```bash
NOCI_INSTALL="$HOME/bin/NOCI.jl/v26.10.0"   # 换成实际已安装且包含本标准的版本。
source "$NOCI_INSTALL/bin/noci-environment.sh"
export PATH="$NOCI_INSTALL/bin:$PATH"
command -v noci-prepare
noci-prepare init "$HOME/noci_cases/ethene"
cd "$HOME/noci_cases/ethene"
```

`init` 复制标准输入、脚本、种子和文档；计算结果写在用户自己的算例目录。
从包含本标准的源码发行包执行 `Install/install.sh` 安装，再加载该版本的环境。
`JULIA_BIN` 指定 Julia 可执行文件，`NOCI_JULIA_PROJECT` 指定所选安装的 Julia 项目，`NOCI_OPENMOLCAS4NOCI` 由安装环境指定 OpenMolcas 启动器。
算例里的 Bash 不保存这些机器路径，部署机器只需设置版本环境与 PATH。

安装内负责解析、计算和数据转换的代码集中于 `scripts/preparation/prepare.jl`、共享 `case_driver.jl`、`standard_manifest.jl` 与 `lib/`，因子转换直接复用 `NOCI.import_seward_hdf5` 和 `NOCI.write_seward_hdf5_products`。
算例目录负责用户输入、运行包装和该算例产物，用户不应从安装内拷贝科学实现到算例目录修改。

## 4. 00 中有哪些设置

先打开 `00_system_define/ethene_dimer.yaml`，里面每个主要区块都有注释。
所有相对文件路径都从所选 YAML 所在目录解析，与调用时的当前工作目录无关。
每个 YAML 是一份完整、独立的定义，不依赖别的 YAML 的隐式继承。

| 区块 | 用户填写的内容 |
| --- | --- |
| `schema_version`、`project`、`method` | 本标准版本 4、唯一输出名、固定 `ahpc_noci`。 |
| `geometry` | 聚集体 XYZ，仅定义各片段目标位置。 |
| `cas` | CAS 后端、轨道基组；OpenMolcas 的单体 RICD/CD 设置，或 PySCF 的 CAS 辅助基。 |
| `caspt2` | 是否计算单态 CASPT2、IPEA／imaginary shift 和最大迭代数。 |
| `templates` | 当前流程的用户输入模板路径，省略时先查找同名本地文件，再使用安装默认。 |
| `molecules` | f/g 等种类、各自点群、载体结构 XYZ、种子引用和逐态 CAS 输入。 |
| `fragments` | f_99 等实例引用哪种分子、哪份载体结构，以及是否自动冻结核心轨道。 |
| `system` | 聚集体总电荷与目标总自旋多重度。 |
| `mebf_table` | 按片段行排列、按 MEBF 列排列的态表。 |
| `common_orbitals` | 当前 project 的轨道并集保留阈值。 |
| `integrals` | 当前 project 的聚集体积分后端、RICD/CD 方法及相应阈值。 |
| `screening` | 对角与非对角 CI 系数筛选阈值。 |
| `runtime` | Julia 线程、BLAS 线程、分块工作区与 GC 提示。 |
| `output` | 是否启用断点。 |

单体 CAS 积分近似与 04 的聚集体积分选择分开声明。
例如 `ethene_dimer_cd.yaml` 的单体仍使用和 RICD 示例相同的 CAS 设置，只将聚集体积分改为 CD，因此原始单体结果可以在共享 01 中自动复用，公共轨道仍分别计算。
如果同时改变单体 CAS 的积分方法或阈值，单体任务定义也会变化，不能沿用原来的同名单体缓存。

OpenMolcas 的 `nactel` 是活性电子数，`inactive` 是双占 inactive 轨道数，`ras2` 是活性轨道数，`multiplicity` 是该片段态的自旋多重度，`ciroot` 与 `rlxroot` 明确指定用户所需的根。
本例 S1 使用 `ciroot: "1 2; 2"`、`rlxroot: 2`，后处理取能量时也保留所选根编号，不按文件名推测根。
这些设置不是自动活性空间建议。

初始轨道支持默认值与逐态覆盖：

```yaml
initial_orbitals:
  default: orbitals/ethene_ccpvdz_seed.RasOrb
  S1: orbitals/ethene_s1special.RasOrb
```

S1 使用专用文件，其他所需态使用 `default`；也可以不写 `default`，逐一列全所需态。
特殊文件由用户提供，标准示例不引用不存在的专用 RasOrb；未使用的默认文件无需读取，拼错的态名会被拒绝。
默认值展开后以实际每态种子进入任务身份，同样文件的简写和完整逐态写法不会改变原始 CAS 任务。

`fragments.f_99.frozen_orbitals: false` 表示不冻结；`true` 表示自动冻结该片段的核心轨道，缺省为 false，不能再写字符串编号。
程序从载体 XYZ 的元素自动求和得到全电子 chemical-core 空间轨道数，采用 [PySCF 的 chemcore_atm 约定](https://pyscf.org/_modules/pyscf/data/elements.html#chemcore)，本地 Julia 表实现，不要求安装 PySCF。
例如乙烯 C2H4 为 2 个核心轨道，丁二烯 C4H6 为 4 个；它们只包括核心，不能把全部 inactive 轨道都冻结。
用户提供的轨道应将核心放在 inactive 块最前面，程序将布尔值展开成 `1-Ncore`，不额外比较各态 inactive 数；轨道顺序及 CAS 设置由用户负责，流程不分析轨道化学性质，也不重排用户活性空间。
当前自动计数针对全电子输入，ECP 扣除及特殊核心选择不在本版契约内；结构审阅模式延后计数，正式检查及运行前必须读取 XYZ 确定实际值。
03 输出实际冻结／删除轨道数，04 自动读取，同一个自动计数贯穿控制文件、波函数与积分处理，不由用户重复填数字。
`common_orbitals.threshold` 缺省为 `1.0e-5`，示例显式写出该值；它与 RICD/CD 积分生成阈值独立。

### 默认开启的 CASPT2 与用户模板

CASPT2 默认开启；省略整个 `caspt2` 区块或其中的 `enabled` 都按 `true` 处理，示例 YAML 也显式开启：

```yaml
caspt2:
  enabled: true
  ipea_shift: 0.0
  imaginary_shift: 0.2
  max_iterations: 80
```

shift 的单位是 Eh，这些示例参数不是对所有体系的科学建议，用户负责选择。
01 在同一单体任务的 RASSCF 后插入单态 CASPT2，默认 `Multistate = 1 <rlxroot>`，明确使用用户选择的根，见 [OpenMolcas CASPT2 手册](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html)。
原始 RasOrb 和 VecDet 在 CASPT2 前导出，波函数沿用原始 CAS 表示；完成记录和 03 的 `.det` 分别保存该根的 CASSCF／CASPT2 能量。
没有 CASPT2 时二者相等，表示零相关修正；最终 NOCI 保留未修正矩阵和能量，并由既有生产接口保存基于片段相关能的修正结果。
启用 CASPT2 或修改其参数会改变共享 01 的任务定义，不能把未做 CASPT2 的旧完成记录当作已完成的新任务。
若不需要 CASPT2，显式设置 `caspt2.enabled: false`。
PySCF 示例同样声明默认开启，但该后端的 CASPT2 和完整工作流适配器尚未实现，本版只支持结构审阅，执行时明确拒绝，不自动跳过 CASPT2。

实际输入模板位于 [00_system_define/templates](00_system_define/templates/README.zh.md)，可以在用户自己的流程中修改。
YAML 明确指定的模板优先；省略 `templates` 时先找 00 下的同名模板，只有没有本地文件才使用安装内 `scripts/preparation/templates` 默认文件。
显式路径缺失会报错，不回退；占位符 `{{UPPER_CASE_KEY}}` 仅作文本替换，OpenMolcas `$Project` 和 `$CurrDir` 保持原样。
OpenMolcas 的 CAS、CASPT2、片段／聚集体参考和积分任务都遵循这条规则，模板修改进入复用判定。
PySCF 也提供原生 DF-CASSCF 和分块 AO DF/CDERI 求解器输入模板，但完整的 bundle、旋转／公共轨道和 NOCI 因子适配尚未实现，结构检查不代表其链路可运行。

### 二聚体与三聚体态表

二聚体的六列依次为 S0S0、S1S0、S0S1、D+D-、D-D+、T1T1：

```yaml
mebf_table:
  - [f_99, S0, S1, S0, D+, D-, T1]
  - [f_03, S0, S0, S1, D-, D+, T1]
```

三聚体载体结构是 GS0S0S0，按用户明确指定的六列输入，而不是另外补一列 S0S1S0：

```yaml
mebf_table:
  - [f_99, S0, S1, D+, D-, T1, T1]
  - [f_03, S0, S0, D-, D+, T1, S0]
  - [f_12, S0, S0, S0, S0, S0, T1]
```

目标总自旋在 `system.spin_multiplicity` 中声明，片段的 T1T1 并不表示把聚集体总自旋也设为三重态。
已有 NOCI 多片段耦合实现处理所允许的自旋耦合；本例每列总单重态有明确的耦合路径。

### 三种因子积分的含义

RICD 使用原子 aCD/acCD 辅助基，`generation_threshold` 控制原子辅助基生成，不能称为分子 ERI 的 Cholesky 截断阈值。
CD 是分子 ERI 的 Cholesky 分解，其 `generation_threshold` 才具有分子 ERI CD 的含义。
DFCD 指未来 PySCF DF/CDERI 的辅助基库仑度量白化因子路线，00 的 `cas.auxiliary_basis` 与 `integrals.auxiliary_basis` 缺省保持一致，只写一处时另一处继承；也可分别明确指定不同值。
至少明确指定一处辅助基，并在 `integrals.metric_lindep` 写出度量参数，不能套用 RICD 或分子 CD 的阈值名称。
三种来源最终都以单位度量 Gram 因子供 AHPC 使用，物理来源及阈值语义应随数据保留。
PySCF 的原生说明可参见 [DF 积分实现](https://pyscf.org/_modules/pyscf/df/incore.html)，本版的实施边界以 [BACKEND_CONTRACT.md](BACKEND_CONTRACT.zh.md) 为准。

## 5. 先检查和规划，不启动计算

在算例根目录执行：

```bash
bash 00_system_define/run.sh ethene_dimer.yaml
noci-prepare plan 00_system_define/ethene_trimer.yaml
noci-prepare plan 00_system_define/ethene2_butadiene.yaml
```

检查包括允许的字段、标签与引用、XYZ 分块和原子顺序、所需种子文件存在、明确的电荷／自旋组合、参数的基本类型与范围。
检查不会分析活性轨道的化学组成或判定 CAS 是否正确，也不会调用 OpenMolcas、PySCF 或 NOCI 求能量。
乙烯二聚体需要 5 个唯一单体任务，对应 10 个实例／态绑定；指定的三聚体同样需要 5 个唯一单体任务，对应 11 个绑定。
f/f/g 示例按其实际态表需要 7 个唯一单体任务；定义但未在态表出现的态不会自动计算。

PySCF 模板只能在本版作结构审阅：

```bash
noci-prepare plan 00_system_define/pyscf_dfcd.yaml --structure-only
```

`--structure-only` 不检查 XYZ/种子文件是否存在，不能代替正式输入验收，也不表示预留后端已经可执行。

## 6. 每一步都是检查、运行、后处理

每个阶段支持相同的模式：

| 模式 | 工作 | 是否可能进行科学计算 |
| --- | --- | --- |
| `prepare` | 检查前置产物、建立任务和输入，写可提交任务清单。 | 不启动量化计算；02 会做小规模几何拟合。 |
| `compute` | 执行本阶段准备好的外部量化任务或最终 NOCI。 | 会。 |
| `finalize` | 检查完成状态、转换数据、发布阶段产物。 | 会进行轨道旋转、CommonOrb 或因子变换等数值后处理，不重启 CAS/SEWARD 求解。 |
| `all` | 按上述顺序完成当前阶段。 | 会。 |

准备、计算和整理的责任相同，耗时数据变换可能位于 `finalize`，不能把 `finalize` 当作完全无计算的文件复制。
队列运行时，可以先 `prepare`，通过阶段目录里的 `tasks_to_submit.txt` 提交尚未完成的 OpenMolcas 输入，外部任务完成后再 `finalize`。
本标准不内置集群、账号、调度器或 SSH，队列提交方式属于部署环境。
05 的 NOCI 是 Julia 任务，应调度对应的 `compute` 命令，不把它当作 OpenMolcas 输入提交。
后续阶段的 `prepare` 需要前一阶段的真实完整产物，不能从空目录一次性准备出所有有效科学输入。

下面命令会启动真实二聚体计算，只读审阅时不要执行：

```bash
bash run_all.sh ethene_dimer.yaml
```

分步执行等价于：

```bash
bash 01_fragment_states/run.sh ethene_dimer.yaml all
bash 02_fragment_placement/run.sh ethene_dimer.yaml all
bash 03_common_orbitals/run.sh ethene_dimer.yaml all
bash 04_integrals/run.sh ethene_dimer.yaml all
bash 05_noci/run.sh ethene_dimer.yaml all
```

需要人工提交 CAS 时，第一步改为：

```bash
bash 01_fragment_states/run.sh ethene_dimer.yaml prepare
# 在计算环境中执行 tasks_to_submit.txt 列出的任务，或使用下行在当前机器直接执行。
bash 01_fragment_states/run.sh ethene_dimer.yaml compute
bash 01_fragment_states/run.sh ethene_dimer.yaml finalize
```

Julia 和 BLAS 线程数在最终 NOCI 启动时分别应用，`scratch_mb` 是分块工作区，不是总内存限制，`heap_size_hint_mb` 是 GC 提示，也不是操作系统内存上限。

## 7. 各阶段的产物如何衔接

### 01：单体计算只做一次

真实任务目录是 `01_fragment_states/f/GS0/S1/` 等，含输入、输出、原始 RasOrb/VecDet、任务定义和完成记录。
`01_fragment_states/bindings/<project>/f_99_S1.RasOrb` 等是该算例的实例引用副本，不触发第二次单体计算。
相同逻辑任务跨二聚体、三聚体 YAML 共用，避免每个编号做一遍相同 CAS。
任务身份直接比较分子／结构／态、分子点群、解析后的坐标和原子顺序、CAS 与启用的 CASPT2 参数、实际输入模板及种子的路径／字节数／纳秒修改时间；小模板还比较完整文本，不对大科学文件计算 SHA。
计算输出按文件存在、大小、修改时间和正常结束状态检查；不读取大文件计算 SHA。
这种轻量身份是复用边界，不能检测人为保持大小和时间戳的内容替换；如用户更换科学输入，应采用新结构标识或明确清理相关任务与下游结果。

### 02：同一变换同时用于 XYZ 和轨道

程序对用户对应的原子做刚性旋转和平移拟合，输出 `matrix_f_99.txt`、`placed_f_99.xyz` 与所有片段重新拼成的 `aggregate.xyz`。
拟合采用用户声明的全原子顺序，输出 RMSD 供用户查看，不自动匹配或交换同元素中心，不以 RMSD 判定载体结构的化学合理性。
XYZ 内部结构始终来自用户提供的单体，聚集体输入只定义目标姿态，程序不会把 RasOrb 当作能够适应几何形变的轨道。
每个位置准备 C1 RunFile/OneInt，再使用相同旋转矩阵产生位置专用轨道。
03、04 和最终几何报告均使用这里的 `aggregate.xyz`，防止轨道已旋转而积分仍在另一份内部结构上生成。

### 03：公共轨道基与短文件名

CommonOrb 使用各片段所需态的轨道并集，产生聚集体 `COMMONORB`，并将片段波函数写成 `f_99_S1.vec`、`f_99_S1.det` 等。
内部文件映射隐藏已有 CommonOrb 工具需要的 project 前缀，最终文件和 NOCI 输入不带这个长前缀。
`.det` 的能量按用户指定根取值，冻结设定来自 00 的布尔选项和自动核心计数，删除计数必须在 CommonOrb 输出中实际找到；缺失时失败，不猜成零。

### 04：只生成因子，不生成四指标 ERI

RICD 输入在 GATEWAY 中写 RICD/CDThreshold，在 SEWARD 中只写 CHH5。
CD 输入不写 GATEWAY RICD，在 SEWARD 写 Cholesky/ThrCholesky/CHH5。
所有输入保持 C1，没有 `&EXPBAS`。
CHH5 到 MO 因子的变换使用实际 COMMONORB 及 03 的冻结／删除计数，生成 `.one`、`.sys` 和 `factors.jld2`；不会生成常规 `.two`，也不会将因子还原成完整四指标矩阵。
`factors.jld2` 是通用因子文件名，不用 `.two.jld2` 来暗示它是一份常规 `.two`。

### 05：输入自动生成，生产结果保留

`05_noci/<project>/<project>.yaml` 自动生成，固定 `algorithm.name: ahpc-noci`、`integrals.backend: factorized`，引用短名波函数和 04 的因子。
不要直接编辑生成文件，变更应回到 00，并使用新的 `project`。
生产运行写 `.sec`、`.out`、最终 `.json`、矩阵快照及可选 checkpoint；`finalize` 要求看到完成事件与最终 JSON。
只有完成真实计算后，才能对矩阵、能量和数值一致性作出验收结论。

## 8. 多 YAML 与共享 01

`project` 只是本版聚集体计算的输出标识，不是复用接口，也不指向别的 YAML。
每份 YAML 自己完整定义体系，其他 project 的名字、执行顺序与当前 project 无关。
同一工作根目录内，只有 `01_fragment_states/` 共享单体计算，其任务定义为分子／载体结构／态及相应 CAS 参数和用户种子。
当前 project 扫描自己所需任务，已完成的同一任务自动复用，没有的任务生成输入并计算，不需要写供体或另一个 project 的名称。

例如乙烯二聚体与三聚体都需要 `f/GS0/S0`、`f/GS0/S1`、`f/GS0/D+`、`f/GS0/D-`、`f/GS0/T1`，这 5 个单体任务各计算一次。
可以先运行二聚体，也可以先运行三聚体，不存在“先完成 ethene_dimer”的要求。
二聚体与三聚体各自完成 02 摆放、03 公共轨道、04 积分与 05 NOCI，不共享聚集体产物。

`ethene_dimer_subset.yaml` 是只包含 3 个中性 MEBF 的独立 project，它所需的单体态为 S0 和 S1。
若共享 01 已完成这两个单体任务就自动复用，否则该 project 自己把缺少的任务算完；它也可以在空工作目录里第一个运行。
其 03 根据自己的态集合构造公共轨道，04 使用自己的实际公共基，既不读取也不保留另一 project 的公共轨道基。
`ethene_dimer_cd.yaml` 同样独立，只将聚集体积分选择设为 CD，原始单体 CAS 定义与 RICD 示例相同时可以自动复用共享 01。

同一 project 的科学输入变更会被拒绝使用自己的旧阶段结果，采用新的 project 可以隔离输出。
共享 `f/GS0/S1` 改了坐标、种子或 CAS 设定则属于共享任务定义冲突，改 project 不会绕过该检查；应使用独立算例根目录或明确清理受影响的单体任务及所有下游计算。
这项检查依据任务输入本身，不需要建立 project 之间的依赖关系。

## 9. 中断与排错

共享 01 中已完成的单体任务会跳过，当前 project 自己的公共轨道和因子产物可用于其中断续算，NOCI 未完成的 checkpoint 由生产接口续算。
完成的 NOCI JSON 不会被原地覆盖，修改态表或筛选设置后应使用新输出定义。
阶段锁通过独占目录创建；出现 `.lock` 时先确认原进程是否仍在运行，只有确认是已退出任务留下的锁才能手工删除。

| 现象 | 要检查的内容 |
| --- | --- |
| 找不到 `noci-prepare` | 是否加载确切安装版本的环境并把其 `bin` 放在 PATH 前面。 |
| 找不到初始轨道 | YAML 路径从 00 解析，所需态是否确有用户种子。 |
| XYZ 分块不一致 | MEBF 行顺序、片段原子数和原子顺序是否与轨道中心对应。 |
| 同名单体任务定义冲突 | 是否换了坐标、基组、CAS 参数或种子却仍引用旧 f/结构/态。 |
| 后续阶段缺产物 | 是否完成前一阶段的 `finalize`；不能用空文件占位启动真实计算。 |
| CommonOrb 没有删除计数 | 查看 CommonOrb 实际输出，不能自己填 0 跳过检查。 |
| RICD CHH5 导出失败 | OpenMolcas4NOCI 版本是否兼容，是否错误地同时写 RICD 与 SEWARD Cholesky。 |
| PySCF 后端尚不可执行 | 本版模板是预留契约，未启用适配器，不要用改后端名字的方式绕过检查。 |

## 10. 给独立审阅者的检查顺序

只读本标准目录即可先核对：00 注释是否足以填写新体系；f/g 与 f_99/f_03/g_05 是否区分清楚；二／三聚体态表是否准确；RasOrb/XYZ 原始共享与逐实例旋转是否一致；共享 01 是否仅包含原始单体态任务；02/03/04/05 是否按 project 独立；RICD/CD 阈值是否分开；05 是否仅 AHPC 与因子积分。
然后检查每个阶段 README 的前置产物、运行动作和输出能否完整衔接，以及所有 Bash 是否仅调用所选安装内的准备程序。
可执行 `check`、`plan` 和 PySCF 的 `--structure-only` 检查，不要执行 `run_all.sh`、`compute` 或数值 `finalize`。
如果要验证准备输入生成，先复制到审阅者自己的临时工作目录；不能在标准源码目录内留下运行产物，模拟产物也不能冒充真实科学验收。
