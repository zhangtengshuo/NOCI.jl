# 安装 NOCI.jl

[English](NOCI_Installation_Walkthrough.md)

本发行适用于 Linux x86_64、glibc 2.28 或更新版本，使用 Julia 1.12.6。
默认安装 CPU NOCI 与 OpenMolcas4NOCI 26.06.2，需要 CUDA 支持时增加 `--withgpu`。
CPU 安装不安装或加载 CUDA。

## 获取源码并查看计划

解压选定源码归档，在源码根目录使用该归档自己的安装器。
解压前核对所提供归档的公开 SHA-256。
`Install/install.sh`、`Install/release.toml` 及支持文件必须来自同一源码修订。
必需工具为 Bash、curl、tar、gzip、sha256sum、realpath 和 flock；准备流程还使用 Python 3。
`Project.toml` 定义 NOCI 版本，`OpenMolcasCompatibility.toml` 定义接受的后端身份。

```bash
bash Install/install.sh --plan
bash Install/install.sh --plan --withgpu
```

计划只打印路径和组件，不写文件或安装软件。
CPU 始终包含在内，没有单独的选择参数。

## 安装

```bash
bash Install/install.sh --execute --prefix "$HOME/bin/NOCI.jl"
# 需要可选 GPU 环境时：
bash Install/install.sh --execute --prefix "$HOME/bin/NOCI.jl" --withgpu
```

对一个安装前缀选择上述其中一条命令执行。
使用 `--julia /path/to/julia` 指定已有 Julia 1.12.6，显式选择的不兼容版本会被拒绝。
未显式指定时，安装器使用 PATH 中兼容的 Julia，或下载已固定的官方归档并核对 SHA-256。
`--source-dir /path/to/source` 选择产品源码，`--manifest /path/to/release.toml` 选择配套安装配置。
`--no-activate` 在成功安装后保留原来的活动版本。
已有版本目录会被保留并拒绝覆盖，独立安装应选择其他前缀。

安装器复制产品文件，实例化精确 Julia 锁，并用独立 micromamba 可执行文件安装固定校验和的原生包。
复制安装文件前会核对所选环境的直接依赖，包括独立准备命令需要的 YAML。
安装使用专用 depot 和明确的项目，不回退到账户默认 Julia 环境。
实例化期间 Project 和 Manifest 保持原摘要，不更新或重新解析依赖。
OpenMolcas 程序前缀与精确原生依赖前缀分离，已完成且兼容的原生前缀可以复用。
激活前检查后端身份、共享库绑定及小型串行 H2 CD/RICD 导出。
显式 GPU 安装还会检查 CUDA 扩展和一个小型 Float64 设备运算。
这些检查全部成功后才写入完整安装记录。

## 安装布局与命令

```text
~/bin/NOCI.jl/
  v26.10.0/                   产品与准备命令
    optional/gpu/             可选 GPU 项目及依赖锁
    bin/noci-environment.sh   生成的版本环境
  noci -> v26.10.0            活动版本
  bin/noci                    版本选择入口
  bin/noci-prepare            准备入口
  OpenMolcas4NOCI/
    v26.06.2-<identity>/       程序前缀
    dependencies/<identity>/  原生依赖前缀
  Julia/
    julia-1.12.6/             需要时下载
    depot/                    管理的包和缓存
  installation/
    records/v26.10.0.toml     完整安装身份
    downloads/                已验证的资产缓存
```

```bash
"$HOME/bin/NOCI.jl/bin/noci" --list
"$HOME/bin/NOCI.jl/bin/noci" --use v26.10.0
"$HOME/bin/NOCI.jl/bin/noci" --version
source "$HOME/bin/NOCI.jl/v26.10.0/bin/noci-environment.sh"
noci-prepare help
```

生成的环境选择该安装的 Julia、项目、depot 和准备命令。
使用后端功能时检查适配；已有兼容积分的计算不会调用后端。
科学运行期间不安装包或解析依赖。
继续阅读 [准备指南](AHPC_NOCI_Preparation_and_Run.zh.md)。

## 可选 GPU 安装

默认安装 CPU NOCI 与 OpenMolcas4NOCI，不安装 CUDA。
增加 `--withgpu` 才安装独立锁定的 GPU 环境，其中包括 CUDA.jl 5.11.0 和 YAML 等准备流程依赖。

```bash
bash Install/install.sh --execute --withgpu
bash Install/install.sh --execute --withgpu --cuda-runtime 12.8
```

未指定 `--execute` 时，安装器只打印计划。
`--cuda-runtime` 为可选参数，必须与 `--withgpu` 一起使用，所选 runtime 需要兼容 NVIDIA 设备与驱动。
CPU 与 GPU 环境各自声明所需依赖，不从账户默认 Julia 环境加载包。
机器专属的 CUDA 偏好只写入安装目录。
安装记录保存所选环境，激活前检查 GPU 扩展和一个小型 Float64 设备运算。

标准计算输入仍默认为 CPU。
在阶段 00 的 manifest 设置 `runtime.use_gpu: true`，阶段 05 才使用已安装的 GPU 环境；`runtime.gpu_device` 或 `runtime.gpu_devices` 选择设备。
Julia 线程数必须为所选 GPU 数量的整数倍。
CAS、轨道和积分准备使用 CPU，阶段 05 子进程保留所选 Julia 环境。
