# RLinf 训练与评估

在交付目录中，先用 `install.sh` 准备任务环境，再用 `rlinf-train` 选择上游脚本并传入 YAML。`example.yaml` 是一份可编辑的完整任务配置。也可以使用包内已经注册的配置名。

## 第一次运行

以下命令适用于默认的 FrankaSim PPO/MLP 示例。请在具备兼容 NVIDIA 驱动的 x86-64 Linux 主机上执行。

```bash
cd release
./install.sh embodied --model openvla --env frankasim
./rlinf-train run_embodiment example.yaml --cfg job
./rlinf-train run_embodiment example.yaml
```

第三行打印 Hydra 组合后的配置并退出，第四行启动训练。若先做一次与下文资源测量设置接近的短实验，可将第四行换成下面的命令。

```bash
GPU=0 RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4000000000 \
  ./rlinf-train run_embodiment example.yaml \
  runner.max_steps=1 runner.val_check_interval=1 runner.save_interval=1 \
  env.train.video_cfg.save_video=false env.eval.video_cfg.save_video=false \
  rollout.enable_torch_compile=false
```

## 选择脚本和配置

命令格式为 `./rlinf-train 脚本名 配置文件或注册名 [Hydra 参数]`。第一项选择运行入口，第二项选择配置。入口与文件名相互独立，因此把文件改名为 `my_sft.yaml` 后仍可明确选择 VLA SFT。

| 用途 | 命令示例 |
| --- | --- |
| 用外置配置训练 | `./rlinf-train run_embodiment example.yaml` |
| 用内置 VLA SFT 配置检查参数 | `./rlinf-train run_vla_sft libero_sft_openpi --cfg job` |
| 用自己的 VLA SFT 配置训练 | `./rlinf-train run_vla_sft ./my_sft.yaml` |
| 选择异步训练脚本 | `./rlinf-train run_async maniskill_sac_mlp_async --cfg job` |
| 选择具身评估脚本 | `./rlinf-train evaluations/run_eval.sh ./my_eval.yaml` |

运行 `./rlinf-train --help` 可查看本包支持的全部脚本，无需先安装依赖。脚本短名唯一时可省略目录和 `.sh` 后缀。多个上游目录都有 `run_eval.sh` 或 `run_train.sh`，请使用帮助中列出的上游相对路径。

提供配置名时，程序查找交付目录的同名 YAML，再查所选脚本的内置配置目录。Embodiment 使用 `examples/embodiment/config/`，VLA SFT 使用 `examples/sft/config/`。显式文件路径支持绝对路径和调用目录下的相对路径。配置可以引用所选入口随包提供的 Hydra 配置组。

## 编辑 example.yaml

`defaults` 选择环境、模型等配置组。`runner` 控制训练步数、日志和检查点，`actor` 配置模型与优化器。可以复制整份文件再修改。只改一个字段时也可以在 YAML 名后加命令行覆盖，例如下面的命令打印改过步数的配置。

```bash
./rlinf-train run_embodiment example.yaml --cfg job runner.max_steps=10
```

默认示例的策略是 `mlp_policy`，安装命令中的 `openvla` 是依赖组。切换模型、模拟器或脚本时，请选用该入口对应的配置，并准备其所需依赖、权重和数据。如果发布者替换了 `example.yaml`，安装参数和脚本名也应按实际任务选择。

## 单机 PPO/MLP 资源起步配置

以下容量适用于单机单卡、状态输入、关闭视频和 rollout Torch 编译的设置。实际训练在 A800-SXM4-80GB（计算能力 8.0）上完成。表中的容量是部署预算，硬件最低配置尚未测定。

| 任务 | 训练 / 评估环境数 | CPU | 主机内存 | GPU | 可用磁盘 |
| --- | ---: | ---: | ---: | --- | ---: |
| ManiSkill PushCube-v1 | 256 / 32 | 8 个逻辑 CPU | 24 GiB | 1 张 NVIDIA GPU，预留 8 GiB 显存 | 25 GiB |
| FrankaSim | 128 / 16 | 8 个逻辑 CPU | 32 GiB | 1 张 NVIDIA GPU，预留 16 GiB 显存 | 25 GiB |

- PushCube 在上述环境规模下完成 3 步训练、评估和保存，采样峰值约为 9.9 GiB 主机进程内存（PSS）和 4.4 GiB 显存。FrankaSim 完成 1 步训练、评估和保存，采样峰值约为 18.1 GiB 主机进程内存和 11.3 GiB 显存，后者按本次任务的 GPU 进程合计。
- 已验证的平台为 Ubuntu 22.04 x86-64、glibc 2.35、CPython 3.11、PyTorch 2.11.0+cu128 和 NVIDIA 驱动 570.172.08。NVIDIA 为 CUDA 12.8 GA 列出的 Linux 对应驱动起点是 570.26，旧版驱动和其他 GPU 尚未通过端到端实测。[NVIDIA CUDA 12.8 发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
- PushCube 的已验证路径使用准备好的依赖环境以及系统 NVIDIA Vulkan ICD。共享库符号核查确认该路径需要可用的 GLIBC 2.34 符号。FrankaSim 的 MuJoCo 模拟器使用 NVIDIA EGL，模拟器图形上下文占用也计入显存预算。Ray 需要可写的短路径临时目录和共享内存。资源短测使用 `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4000000000`，按同一配置部署时预留至少 4 GiB 可用 `/dev/shm`。
- 测试环境安装后占约 8.9 GiB，另有约 74 MiB 可执行文件和 182 MiB 解包缓存。25 GiB 预算为环境导入、临时文件及该规模的日志和检查点留出余量。在线重新安装、依赖编译、模型权重和数据可能需要额外空间。
- 调整环境数、批量大小、输入类型、视频、模型或 GPU 数量后，按实际配置重新测量。完整 OpenVLA 和 SFT 同样需要单独测量。

## 安装、日志与迁移

安装需要 Bash、`flock`、Git、curl、wget 以及任务依赖的系统库。联网安装通过内置的上游依赖清单选择具身模型与模拟器。仅运行默认 MLP 示例时，也可用 `./install.sh embodied --model openvla --env frankasim --task frankasim_ppo_mlp` 选择锁定依赖，此方式还需要 `dpkg-deb`、GCC 和 G++。其他任务如果已有装好依赖的可迁移独立 CPython 3.11 环境，可以用 `./install.sh --prepared-runtime /path/to/runtime` 导入。安装选项见 `./install.sh --help`。

每次运行会打印日志目录，训练日志和检查点保存在交付目录的 `logs/` 下。恢复训练可设置 `runner.resume_dir`。使用默认训练 YAML 做独立评估时，设置本地权重路径，并补全评估入口需要的模型配置。

```bash
CHECKPOINT='/absolute/path/to/full_weights.pt' GPU=0 ./rlinf-train evaluations/run_eval.sh example.yaml \
  'rollout.model=${actor.model}' 'runner.ckpt_path=${oc.env:CHECKPOINT}'
```

首次解包和安装会在本目录生成 `.rlinf-app/`、`.envs/` 和 `.venv` 等隐藏项。迁移已安装环境时，使用 `cp -a` 等方式复制整个目录，保留符号链接。检查 YAML 中的权重和数据路径，并确保目标主机的驱动和系统库兼容。

## 在 v0.1 和源码版之间续训

当两边使用相同上游版本、任务配置和兼容的 PyTorch 环境时，具身 PPO 的 FSDP 检查点可以通过 `runner.resume_dir` 交接。传入 `global_step_N` 目录，保留其中的 `actor/` 和完整 DCP 文件。此入口恢复模型、优化器及调度器。单独的 `full_weights.pt` 适用于权重加载或评估。

v0.1 修正了 FSDP 的 LambdaLR 回调在 Cython 下的序列化。早期交付版写出的检查点可先用 v0.1 续训一步，保存到新目录后再交给源码版。例如，若旧任务已训练 300 步，请沿用它的 YAML 和任务环境。

```bash
./rlinf-train run_embodiment ./my_task.yaml \
  runner.resume_dir=/path/to/old/checkpoints/global_step_300 \
  runner.logger.log_path=./logs/migrated runner.max_steps=301 runner.save_interval=1
```

将新日志目录下的 `checkpoints/global_step_301` 传给源码版的 `runner.resume_dir`，并将目标步数设为大于 301。这个迁移例子会执行一次真实参数更新。
