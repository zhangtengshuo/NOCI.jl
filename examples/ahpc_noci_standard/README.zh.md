# 推荐的 AHPC-NOCI 通用标准流程

从 [TUTORIAL.md](TUTORIAL.zh.md) 开始，再阅读 [00_system_define/ethene_dimer.yaml](00_system_define/ethene_dimer.yaml) 中的注释。
本发行采用 `schema_version: 4`，仅提供这个标准算例目录。
本版提供 OpenMolcas CAS → OpenMolcas RICD/CD → AHPC-NOCI 的 C1 标准路线。
准备阶段使用 CPU；阶段 05 默认使用 CPU，也可在显式安装 GPU 环境后以 `runtime.use_gpu: true` 启用加速。
各分子在 `molecules.<id>.symmetry` 声明自己的点群，本版只执行 C1；OpenMolcas 01 默认计算单态 CASPT2，设 `caspt2.enabled: false` 可关闭。
用户可编辑的实际量化输入模板位于 [00_system_define/templates](00_system_define/templates/README.zh.md)，流程内文件优先，安装默认只作兜底。
PySCF DFCD 的辅助基、种子格式和后端接口见 [BACKEND_CONTRACT.md](BACKEND_CONTRACT.zh.md)，对应模板本版仅可作结构检查。

一个分子的多套态波函数也可进行 NOCI，见 [乙烯单分子定义](00_system_define/ethene_monomer.zh.yaml) 和详细教程的单分子章节。

```text
00_system_define/       多个独立 project 的完整 YAML、XYZ 和用户种子
  templates/            当前流程的可编辑实际输入模板
    openmolcas/         cas.in、caspt2.in、fragment_reference.in、aggregate_reference.in
                        integrals_ricd.in、integrals_cd.in
    pyscf/              cas.py.in、integrals_dfcd.py.in（后端适配预留）
01_fragment_states/     唯一共享阶段：f/GS0/S1 等单体 CAS 任务
02_fragment_placement/  <project>/：该 project 的摆放和旋转轨道
03_common_orbitals/     <project>/：该 project 的公共轨道和 .vec/.det
04_integrals/           <project>/：该 project 的 .one/.sys/factors.jld2
05_noci/                <project>/：该 project 的 NOCI 输入、结果和断点
```

不同 project 不需要知道彼此的名字或执行顺序，只按共同的分子／结构／态定义自动复用 01 单体结果。
01 中已有完成任务就复用，缺少的任务才计算；02、03、04、05 均按当前 project 独立生成，不从另一个 project 复制数据。
`f` 表示同一种分子，`f_99`、`f_03` 是不同位置编号；MEBF 行顺序决定片段顺序，编号不排序；标签允许任意字母数字串_字母数字串，详见教程。
同分子、同结构、同态共享旋转前 RasOrb 和 CI，之后各实例按自己的位置旋转。
活性空间与 RasOrb 的科学正确性由用户负责，流程不分析或选择活性空间，不进行正常运行中的全文件 SHA 校验。

算例 Bash 只定位当前 YAML 并调用 PATH 中所选版本的 `noci-prepare`，计算实现统一维护在安装位置。
只检查而不计算：

```bash
noci-prepare check 00_system_define/ethene_dimer.yaml
noci-prepare plan 00_system_define/ethene_trimer.yaml
```

`run_all.sh` 会启动真实计算，审阅时不要执行。
