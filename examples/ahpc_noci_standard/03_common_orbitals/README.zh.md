# 03：当前 project 的公共轨道与片段波函数

运行前：确认当前 project 的 02 摆放产物，生成控制和文件映射，按用户的 true/false 选项自动确定核心冻结数。
运行：准备当前聚集体的 C1 RunFile/OneInt。
运行后：调用安装内 CommonOrb，按当前 project 所需态构造公共基，生成 COMMONORB、短名 f_99_S1.vec/.det 和冻结／删除计数摘要。

输出全部位于当前 `<project>/`，与其他 project 的公共轨道没有关系，也不引用其他 project 的名字。
态表的轨道并集由当前 project 决定，子集 project 正常构造自己的公共轨道，不为了复用另一个 project 而保留其基。
同一个 project 中断后可以继续使用自己的完整产物，但不同 project 不共享 01。

```bash
bash run.sh ethene_dimer.yaml all
```
