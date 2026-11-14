# 00：定义体系

选择一个 YAML 作为本版计算的唯一权威输入，先填分子种类及各自点群、载体 XYZ、用户种子、CAS 参数、片段编号和 MEBF 态表，再填积分与运行设置。
全部相对路径从该 YAML 所在目录解析，多个 YAML 可以共用这里的 XYZ 和种子，且必须为不同输出使用不同 `project`；各 project 不引用其他 project。

| 文件 | 用途 |
| --- | --- |
| `ethene_monomer.yaml` | 一个乙烯分子，S0／S1 两个 singlet MEBF，只有一行片段定义。 |
| `ethene_dimer.yaml` | GS0S0；S0S0、S1S0、S0S1、D+D-、D-D+、T1T1。 |
| `ethene_trimer.yaml` | GS0S0S0；严格采用用户指定的六列三聚体态表。 |
| `ethene2_butadiene.yaml` | f_99/f_03 为乙烯，g_05 为丁二烯，展示 f/f/g 异种体系。 |
| `ethene_dimer_subset.yaml` | 独立的三个中性 MEBF project；01 可自动复用同名单体任务，03/04 独立生成。 |
| `ethene_dimer_cd.yaml` | 独立 CD project；单体 CAS 定义不变，01 可自动复用，后续均独立生成。 |
| `pyscf_dfcd.yaml` | PySCF 原生种子与显式辅助基的预留接口模板，不能启动计算。 |

```bash
bash run.sh ethene_dimer.yaml
noci-prepare plan 00_system_define/pyscf_dfcd.yaml --structure-only
```

`run.sh` 只做输入检查，没有真实计算。
用户应提前完成需要的结构优化和频率计算，这些任务不在 00／01／02／03／04／05 流程中。

`caspt2` 默认开启，OpenMolcas 按所选根计算单态 CASPT2，设 `enabled: false` 可关闭；PySCF CAS 的辅助基在 `cas.auxiliary_basis` 指定，缺省与聚集体 DFCD 的辅助基一致。
[templates/](templates/README.zh.md) 是用户可修改的实际输入模板，所有模板选择都优先当前流程文件，缺少本地默认文件时才回退安装内模板。

各 `.zh.yaml` 是相同科学定义的中文注释版本，默认 `.yaml` 使用英文注释，计算时选择其中一个即可。
