# CAS／积分后端契约与扩展边界

本标准将 `cas.backend` 与 `integrals.backend` 分开声明，工作目录和 00／01／02／03／04／05 含义不依赖某个量化程序的名字。
OpenMolcas4NOCI v26.06.2 的 CHH5 和已有 CommonOrb／RotateOrb 是本版可执行实现；PySCF 原生 DFCD 是后续适配器，不在本版声称完成的数值范围内。

| CAS 后端 | 积分后端／方法 | 本版状态 |
| --- | --- | --- |
| OpenMolcas | OpenMolcas／RICD | 可执行准备链，真实计算尚未在新标准上验收。 |
| OpenMolcas | OpenMolcas／CD | 可执行准备链，真实计算尚未在新标准上验收。 |
| PySCF | PySCF／DFCD | schema、辅助基、种子和产物契约预留；可作结构检查，执行前明确拒绝。 |
| OpenMolcas | PySCF／DFCD | 后续混搭扩展，本版在输入检查时拒绝。 |
| 任意 | exact 四指标 ERI | 不属于本推荐标准。 |

## 稳定的阶段接口

各适配器遵循 `prepare`、`compute`、`finalize`，保留同样的分子／结构／实例定义和阶段目录。
实现集中于安装内：`STANDARD_BACKENDS` 声明受支持组合与能力，`PREPARATION_STAGE_ADAPTERS` 注册该组合的阶段处理函数。
增加后端时添加对应实现与能力注册，不复制一套算例 Bash，阶段目录不随后端另起一套名字，也不让用户重排 00／01／02／03／04／05。
分子点群在各自 `molecules.<id>.symmetry` 声明；本版仅执行 C1，非 C1 点群只保留结构审阅接口。
以后混搭需增加 AO 数据桥接适配器和能力组合，当前的 `cas.backend`／`integrals.backend`、片段身份、路径布局与阶段职责不需要改变。
参数校验和后端专有选项可以随新适配器增加，不能把“骨架不变”解释为无需实现或验证新的数值功能。

| 阶段 | 后端适配器必须提供的结果 |
| --- | --- |
| 01 | 与用户输入种子、载体 XYZ、基组和态定义关联的原始轨道、CI、能量和后端元数据；共享任务按分子／结构／态定义。 |
| 02 | 实例专用位置变换、旋转后的轨道、摆放 XYZ、按固定实例顺序拼合的 aggregate XYZ，保留原始单体数据。 |
| 03 | 实际保留公共轨道基与各实例／态 `.vec/.det`；显式的基定义、冻结信息和必要的后端维度信息。 |
| 04 | `integrals.one`、`integrals.sys`、`factors.jld2`；一体算符与因子必须在同一实际公共轨道基上。 |
| 05 | 固定 AHPCNOCI 与 `factorized` 输入，仅引用短实例／态标签；后端差别不进入算法分派。 |

## PySCF 原生 CAS 的输入边界

PySCF 的种子是其自身 AO 约定下的 MO 系数，预留模板使用 `.npy` 路径，适配器还应记录配套的 AO 中心、基组、球谐／Cartesian 约定、系数方向及顺序元数据。
种子由用户提供，不用 AVAS、自动排序或自动选择活性空间替代用户的轨道定义。
`spin_2S`、`nelecas: [n_alpha, n_beta]`、`ncore`、`ncas` 与一开始编号的 `root` 保留原 PySCF 标准的物理含义。
`ncore` 是 CAS 中的双占 inactive 数，不能被当作 NOCI 冻结轨道数。
原 `NOCI_PySCF_FR_standard_20260809` 的状态 bundle、旋转、矩形 CommonOrb 和 determinant 导出是后续迁移参考；不能将其 exact 四指标积分生成部分或自动活性策略直接带入本标准。
矩形公共轨道表示不应虚构 OpenMolcas 的 deleted 计数，03 的通用接口应承载实际 AO 维度、保留轨道维度与明确冻结子空间。
本版没有提供这些 Python 适配器、PySCF 数值测试或用户 `.npy` 种子，不能把模板当作已跑通案例。

## DFCD：显式辅助基与度量白化

DFCD 在这里特指 PySCF DF/CDERI 因子路线，使用三中心积分与辅助基库仑度量，度量白化后得到单位度量 Gram 因子。
其含义与 conventional molecular ERI CD 不同，也与 OpenMolcas 的原子 RICD 不同。
`cas.auxiliary_basis` 与 `integrals.auxiliary_basis` 缺省一致，只填一处则另一处继承，允许分别显式指定不同值；至少填写一处辅助基，并在 `integrals.metric_lindep` 写明度量处理参数；不让程序自动挑选辅助基或用分子 CD 阈值解释该参数。
后续实现需按辅助维度分块把 AO 因子变换到公共轨道基，保留 pair 行 × auxiliary 列布局，不形成完整 pair × pair ERI，不生成 `.two`。
冻结核心的一体有效算符与常数能量必须由一致的因子和冻结子空间构造，不能只删除 CI 电子却忽略有效 Hamiltonian 修正。
输出必须新增明确 DFCD 来源及辅助基名称、度量处理方法／阈值／单位的元数据契约，不得将当前仅支持 conventional CD/RICD 来源的 `ERIFactorProvenance` 伪装成 DFCD 已验收。
数值 kernel 可以共享 `FactorizedERI` 表示，但新来源元数据与读写兼容性需要单独实施与测试。

## 未来 OpenMolcas CAS → PySCF DFCD 混搭

混搭适配器应在 02／03 衔接处将 OpenMolcas 原始 AO 约定转换为 PySCF 原生 AO 表示，再用相同的实际公共轨道基生成一体算符和 DFCD 因子。
AO 原子顺序、收缩、归一化、实球谐顺序、符号和轨道系数方向必须有确定的数据映射及数值闭合测试，不能凭两边基组字符串同名就判定兼容。
这属于数据表示兼容检查，不是替用户分析活性空间的正确性。
用户仍提供原始轨道及 CAS 设定，程序不能为混搭自动选择新的活性空间。
01 的共享原始任务身份取决于分子、结构、态、CAS 配置及用户种子，不因后来换积分后端而重复做单体 CAS；当前 project 的 03／04 续算检查必须包含其实际公共基与后端定义。
完成适配器、来源元数据、维度／重构测试和真实小体系矩阵验收后，才能把相应能力从预留改为可执行。

跨 project 只共享 01 的原始单体态任务；02／03／04／05 的适配器都处理当前 project 自己的数据，不注册供体 project 或共享聚集体结果。

## 可选能量与实际输入模板

OpenMolcas 01 默认开启单态 CASPT2，可通过 `caspt2.enabled: false` 关闭，并把所选根的能量传给 03；PySCF CASPT2 能力尚未实现，不能隐式替换成其他 PT2 方法。
未计算 CASPT2 时在新标准 `.det` 中记录与 CASSCF 相同的能量，产生零相关修正；原始波函数不因 CASPT2 能量计算改变。
所有 OpenMolcas 量化任务选择用户模板后渲染；PySCF 提供原生求解器输入模板，完整后端仍 reserved。
用户模板、分子点群和启用的 CASPT2 参数进入任务来源与复用条件，流程文件优先于安装默认，不引入 project 间依赖。
