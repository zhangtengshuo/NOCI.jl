# schema-4 AHPC-NOCI 准备与运行

[English](AHPC_NOCI_Preparation_and_Run.md)

完整输入和科学解释见 [标准算例详细教程](../../examples/ahpc_noci_standard/TUTORIAL.zh.md)。
本指南使用同一套 schema-4、C1 CPU/OpenMolcas、所选根 CAS/CASPT2 和 RICD/CD 因子积分流程。
先按 [安装指南](NOCI_Installation_Walkthrough.zh.md) 安装 CPU 发行版。

## 复制算例

```bash
NOCI_INSTALL="$HOME/bin/NOCI.jl/v26.10.0"
source "$NOCI_INSTALL/bin/noci-environment.sh"
noci-prepare init "$HOME/noci_cases/ethene"
cd "$HOME/noci_cases/ethene"
noci-prepare check 00_system_define/ethene_dimer.yaml
noci-prepare plan 00_system_define/ethene_dimer.yaml
```

检查前修改用户副本中的 YAML、XYZ、种子和模板。
核对单体结构、基组、AO 顺序和所选根；程序不选择或分析活性空间。
正式 check/plan 检查后端身份，但不启动计算。
无后端或文件时可用 `--structure-only` 审阅 schema；它不证明预留 PySCF 路线可执行。

## 执行各阶段

以下命令会启动真实计算：

```bash
bash run_all.sh ethene_dimer.yaml
```

队列执行时逐阶段使用 `prepare`，根据 `tasks_to_submit.txt` 提交尚未完成的任务，再对真实完成的结果执行 `finalize`。
当前机器直接计算使用 `compute`，完成当前阶段全部动作使用 `all`。
05 是 Julia 作业；finalize 中的旋转、CommonOrb 和积分变换也会执行数值工作。

| 阶段 | 必需产物 |
| --- | --- |
| 01 states | 原始 RasOrb/VecDet、所选根能量、完成记录和后端身份。 |
| 02 placement | 同一位置变换下的 XYZ 与实例专用旋转轨道。 |
| 03 commonorb | 实际 COMMONORB、短名 `.vec/.det`、冻结/删除计数。 |
| 04 integrals | 实际公共基下的 `integrals.one`、`integrals.sys`、`factors.jld2`。 |
| 05 run | 自动 AHPC/factorized YAML、进度、`.out`、最终 `.json`、快照和可选断点。 |

只有原始 01 单体任务跨 project 共享，聚集体公共基、积分和结果各自独立。
RICD 原子辅助基阈值与分子 CD 阈值含义不同。
CASPT2 默认开启，可以显式关闭；关闭时片段相关修正为零，CAS 波函数保持原样。
本标准不生成常规四指标 `.two`。

## 续算和变更

复用要求科学定义、种子、模板与后端身份一致，且有真实有效产物。
聚集体输入改变时使用新 project；同名单体任务改变时使用新载体标识或独立算例根，不能只改 project 绕过冲突。
未完成 NOCI 用有效生产断点续算，不能用改变后的计算覆盖原最终 JSON。
删除陈旧锁之前先检查其所属进程。
队列、账号和节点临时目录属于部署配置，不写入标准算例。
冻结核心假设、所选根、MEBF/自旋定义和排错见 [详细教程](../../examples/ahpc_noci_standard/TUTORIAL.zh.md)。
