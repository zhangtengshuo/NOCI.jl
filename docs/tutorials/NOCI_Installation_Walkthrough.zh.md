# 安装 CPU 发行版

[English](NOCI_Installation_Walkthrough.md)

本发行支持 Linux x86_64、glibc 2.28 或更新版本，并固定 Julia 1.12.6。
本版不提供 GPU 安装或执行；默认安装 CPU NOCI 与 OpenMolcas4NOCI 26.06.2。
已有兼容积分和波函数输入时可以使用 `--with cpu`。
以下安装命令只有带 `--execute` 时才下载、写环境并运行小型安装检查。

## 1. 获取源码

下载选定 tag 对应的 NOCI 源码或发行归档，解压到用户自己的目录。
使用专门发行归档时，先与公开校验和核对 SHA-256，再解压。
始终配套使用该源码的 `Install/install.sh`、`Install/release.toml` 与支持文件，不能混用安装器版本。
安装器默认读取自身所在源码目录，无需猜测 NOCI 仓库地址。
系统需提供 Bash、curl、tar、sha256sum 和 flock；现有准备工具还使用 Python 3。
`Project.toml` 是 NOCI 版本权威，`OpenMolcasCompatibility.toml` 记录后端适配。

## 2. 查看计划

在解压后的源码根目录执行：

```bash
bash Install/install.sh --plan
bash Install/install.sh --plan --with cpu
```

这些命令只显示路径和组件，不下载、不写文件、不安装、不测试。
`--with openmolcas` 和 `--all` 都选择 CPU 加 OpenMolcas；请求 GPU 组件会失败。

## 3. 安装

```bash
bash Install/install.sh --execute --prefix "$HOME/bin/NOCI.jl"
```

使用 `--julia /path/to/julia` 指定已有 Julia 1.12.6，或用 `--source-dir /path/to/extracted/source` 指定另一个源码目录。
没有合格的选定 Julia 时，安装器下载固定官方归档并核对 SHA-256。
`--no-activate` 表示安装成功后仍保留原来的活动版本。
已有版本目录会被拒绝覆盖；应检查未完成安装，而不是直接覆盖。

安装器检查源码与配置版本、CPU 依赖，按产品清单复制，实例化精确 Julia 锁，并安装固定校验和的原生资产。
OpenMolcas 程序与原生依赖放在不同前缀；经过检查且身份一致的依赖可以跨版本复用。
独立 micromamba 按精确锁安装原生包，不升级所选科学依赖版本。
激活前核对程序身份、库绑定、源码 revision 和声明的后端能力。
获准安装时会执行 H2 的 CD 和 RICD/CHH5 小型检查；这些检查不代表完整 CAS/CommonOrb/NOCI 算例已验收。
仅成功安装才写完成记录并按选项切换活动版本。
失败保持原活动版本，并保留未完成目录供检查。

## 4. 安装布局

```text
~/bin/NOCI.jl/
  v26.10.0/                   CPU 产品及该版本自己的准备入口
  noci -> v26.10.0             活动版本
  bin/noci                    版本选择入口
  OpenMolcas4NOCI/
    v26.06.2-<identity>/      固定程序前缀
    dependencies/<identity>/ 共享原生依赖前缀
  Julia/
    julia-1.12.6/             需要时下载
    depot/                   共享 Julia 包和缓存
  installation/
    records/v26.10.0.toml     完成安装身份
    downloads/               已校验资产缓存
```

纯 CPU 安装省略 OpenMolcas 组件，运行已有兼容数据时不需要安装后端。
科学运行期间不安装、求解或下载依赖。

## 5. 选择确切版本

```bash
"$HOME/bin/NOCI.jl/bin/noci" --list
"$HOME/bin/NOCI.jl/bin/noci" --use v26.10.0
"$HOME/bin/NOCI.jl/bin/noci" --version
NOCI_INSTALL="$HOME/bin/NOCI.jl/v26.10.0"
source "$NOCI_INSTALL/bin/noci-environment.sh"
command -v noci-prepare
```

版本环境把自己的 bin 放在 PATH 中，准备程序使用这个版本的实现和记录的后端，不设置另一个根级准备入口。
正式准备检查将实际后端身份与兼容配置比较；未知版本即使数字更大，也会被拒绝。
然后阅读 [schema-4 准备指南](AHPC_NOCI_Preparation_and_Run.zh.md)。

## 6. 安装排错

先检查空间、固定资产网络访问、Julia 版本和必需工具。
校验和失败不能绕过，应取得正确的固定资产。
后端与库不匹配时，应选用同一套记录的程序、根目录和依赖环境。
不能用普通上游 OpenMolcas 替代声明的 OpenMolcas4NOCI 构建。
对于未完成安装，保留证据并选择独立前缀重新尝试。
本指南中的命令不构成当前发行已经通过目标机器验收的声明。
