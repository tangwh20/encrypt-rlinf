# RLinf 编译交付工具

用 Cython 和 GCC 编译上游 RLinf 的业务模块，生成可以单独交给使用者的 Linux x86-64 代码包。训练与评估共用一个可执行文件，运行者通过第一个参数选择脚本，通过第二个参数选择 YAML 或内置配置名。

v0.1 以原生扩展交付训练代码。收件人仍可查看配置、模块名和运行时脚本文本，也可通过调用原生函数推断算法。构建会去除内嵌评测脚本中的普通注释，保留其执行逻辑、文档字符串及诊断行号。

## 构建

在本仓库根目录克隆一份 RLinf，再运行编译脚本。以下示例使用已验证的上游提交。

```bash
git clone https://github.com/StridingAI/RLinf-Striding.git RLinf
git -C RLinf checkout --detach bf0e1664216a7d640abfb7c841fee43f90a9fba0
./compile.sh
```

编译器会识别根目录中的上游检出，从中读取代码、配置、许可证和安装配方。默认输出如下。

```text
release/
├── rlinf-train
├── install.sh
├── example.yaml
└── README.md
```

`rlinf-train` 包含原生模块、配置组和安装所需的配方。`install.sh` 在目标机器准备任务环境。`example.yaml` 是可编辑的完整配置，默认取自已验收的 FrankaSim PPO/MLP 任务。如果上游版本缺少该配置，构建器会选用具身入口的默认 ManiSkill 配置或该目录中的首份 YAML。目录内的其他注册配置同样可以按名使用。

可以将另一份完整配置作为第二个位置参数，构建器会将其原样发布为 `example.yaml`。输出目录必须是新目录。如果根目录中有多个上游检出，可用 `--source` 指定。

```bash
./compile.sh release-sft RLinf/examples/sft/config/libero_sft_openpi.yaml --source RLinf
```

默认构建会联网下载独立 CPython 3.11 和 Cython。构建机需有 x86-64 Linux、Python 3.10 或更高版本、Bash、GCC、GNU 工具、curl、dpkg-deb，以及 glibc 和 zlib 的静态开发库。Ubuntu 可通过 `libc6-dev` 和 `zlib1g-dev` 提供这两项开发库。

## 使用

把整个 `release/` 复制到目标机器。在默认示例上，可用以下命令安装环境、查看组合后的配置并启动训练。

```bash
cd release
./install.sh embodied --model openvla --env frankasim
./rlinf-train run_embodiment example.yaml --cfg job
./rlinf-train run_embodiment example.yaml
```

使用 VLA SFT 时，选择 `run_vla_sft` 并提供该任务的 YAML 或内置名字。其他入口可运行 `./rlinf-train --help` 查看。每类任务需要相应的依赖、权重和资源。单机 FrankaSim PPO/MLP 已完成训练与评估验证，完整 OpenVLA 策略栈仍有上游依赖冲突。原生编译可避免直接分发 RLinf 的 Python 业务源码，二进制仍可被逆向分析。配置编辑、脚本选择和迁移见[交付使用说明](delivery/RELEASE.md)。

## 单机资源起步配置

以下为状态输入、关闭视频和 rollout Torch 编译的单卡 PPO/MLP 配置。训练和评估环境数与实测长跑一致。显存、内存和磁盘容量是给首次部署预留的预算，尚未测得端到端的最低硬件配置。

| 任务 | 训练 / 评估环境数 | CPU | 主机内存 | GPU | 可用磁盘 |
| --- | ---: | ---: | ---: | --- | ---: |
| ManiSkill PushCube-v1 | 256 / 32 | 8 个逻辑 CPU | 24 GiB | 1 张 NVIDIA GPU，预留 8 GiB 显存 | 25 GiB |
| FrankaSim | 128 / 16 | 8 个逻辑 CPU | 32 GiB | 1 张 NVIDIA GPU，预留 16 GiB 显存 | 25 GiB |

PushCube 短测完成 3 步训练、评估与保存，采样峰值约为 9.9 GiB 主机进程内存（PSS）和 4.4 GiB 显存。FrankaSim 完成 1 步训练、评估与保存，采样峰值约为 18.1 GiB 主机进程内存和 11.3 GiB 显存。关闭视频时 MuJoCo 仍会分配 NVIDIA EGL 图形上下文。

从 Ubuntu 22.04 x86-64（glibc 2.35）和兼容 CUDA 12.8 的 NVIDIA 驱动环境起步。实测驱动为 570.172.08，实测 GPU 为 A800-SXM4-80GB（计算能力 8.0）。PushCube 还需要可被 Vulkan loader 发现的 NVIDIA Vulkan ICD，实测表明该训练路径需要 GLIBC 2.34 对应的符号。采用短测的 Ray 对象存储上限 4,000,000,000 字节时，建议至少预留 4 GiB 可用 `/dev/shm`，并提供可写的短路径临时目录。其他 GPU、驱动版本、容器以及视觉、视频、完整 OpenVLA 或 SFT 任务需要按实际配置重新测量。短测数据与安装细节见[交付使用说明](delivery/RELEASE.md)。
