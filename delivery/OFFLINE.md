# v0.2 离线训练、评估与策略部署

交付目录包含 `run`、`install.sh`、`example.yaml` 和本说明。将四个文件复制到目标主机的可写目录，保留可执行权限。安装器从 `run` 读取并校验内嵌运行时，安装过程无需网络、系统 Python、外部依赖缓存或源码仓库。

发行包如包含自动选中的原生模块，启动器在首次创建应用缓存时会验证并解密这些模块。后续启动复用缓存中的原生扩展。若缓存文件受损，可在所有任务退出后将 `.train-app` 改名，再次启动即可重新验证并生成缓存。包含加密模块的启动器使用 OpenSSL libcrypto（Copyright 1995–2020 The OpenSSL Project Authors），采用 Apache License 2.0，可用 `./run --runtime-dir` 定位应用目录，其中的 `app/LICENSE` 提供 Apache License 2.0 全文。

## 安装并启动

在具备兼容 NVIDIA 驱动的 x86-64 Linux 主机上执行。安装需要 Bash、`flock`、`tar`、`gzip` 及标准 GNU 工具。

```bash
./install.sh --offline
./run embodiment example.yaml --cfg job
GPU=0 ./run embodiment example.yaml
```

安装命令可以省略 `--offline`。第二条命令打印 Hydra 组合配置，第三条启动训练，使用物理编号为 0 的 GPU。默认 `example.yaml` 是 FrankaSim PPO/MLP 任务。若希望先运行一次训练、评估和保存，可使用以下设置。

```bash
GPU=0 RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4000000000 \
  ./run embodiment example.yaml \
  runner.max_steps=1 runner.val_check_interval=1 runner.save_interval=1 \
  env.train.video_cfg.save_video=false env.eval.video_cfg.save_video=false \
  rollout.enable_torch_compile=false
```

若发布方随包提供了数据或权重，`./install.sh` 会把它们解包到当前离线环境的 `assets/`。YAML 可用 `${oc.env:TRAIN_OFFLINE_ASSETS}/model.safetensors` 引用该目录中的文件，也可以按相同写法引用数据文件。将这个路径填在任务实际使用的模型或数据配置字段中。安装与启动会使用同一份本地环境，无需单独拷贝随包资产。

## 选择任务与配置

命令格式为 `./run 入口名 配置文件或注册名 [Hydra 参数]`。用 `./run --help` 查看入口及兼容的脚本路径。也可以用原入口名 `run_embodiment` 或完整路径 `examples/embodiment/run_embodiment.sh`。可传入外部 YAML 文件的相对或绝对路径，也可以使用注册名，例如 `./run embodiment maniskill_ppo_mlp --cfg job`。配置文件可用 `defaults` 引用已打包的配置组，也可用命令行参数覆盖字段。

随包运行时包含 FrankaSim、MuJoCo、ManiSkill、SAPIEN/PhysX 及 ManiSkill 的 Panda 机器人本地资源。适用范围为默认 FrankaSim PPO/MLP 与使用 Panda 的 ManiSkill PushCube-v1 PPO/MLP。两类任务的 MLP 策略可以从随机初始化开始。其他场景、机器人、OpenVLA 或 VLA SFT 所需的模型、数据、资产和依赖需按任务另行准备。检查自行提供的 YAML 中的模型、数据及检查点路径。

用内置的 `maniskill_ppo_mlp` 运行一次 PushCube 训练和评估时，将环境切换为 PushCube-v1，并将策略输入维度设为 35。

```bash
GPU=0 RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4000000000 \
  ./run embodiment maniskill_ppo_mlp \
  env.train.init_params.id=PushCube-v1 env.eval.init_params.id=PushCube-v1 \
  ++env.train.init_params.render_backend=none ++env.eval.init_params.render_backend=none \
  env.train.total_num_envs=256 env.eval.total_num_envs=32 \
  actor.model.obs_dim=35 actor.micro_batch_size=1600 actor.global_batch_size=1600 \
  env.train.video_cfg.save_video=false env.eval.video_cfg.save_video=false \
  rollout.enable_torch_compile=false rollout.enable_cuda_graph=false \
  runner.max_steps=1 runner.val_check_interval=1 runner.save_interval=1
```

## 系统与磁盘

参考平台为 Ubuntu 22.04 x86-64、glibc 2.35、NVIDIA A800 和驱动 570.172.08。PushCube GPU 仿真依赖系统上的 NVIDIA CUDA 驱动和 Vulkan ICD。FrankaSim 的 MuJoCo 使用 NVIDIA EGL。Ray 需要可写的短路径临时目录和共享内存。按上面的短测配置运行时，预留至少 4 GiB 可用 `/dev/shm`。

安装还需约 9.1 GiB 存放解压后的运行时及 PhysX GPU 库，另有应用缓存、训练日志和检查点。建议在复制交付文件后为安装及短测保留约 25 GiB 可用空间。实际训练资源随环境数、模型、输入类型和保存频率而变化。

## 迁移与检查点

安装在交付目录中创建 `.envs/`、`.venv` 和 `.train-app/`。可以只复制四个交付文件后在新路径重新离线安装，或通过 `cp -a` 复制已安装目录以保留隐藏文件和符号链接。更换机器后请检查系统驱动和外部数据路径。

训练输出在 `logs/`。继续训练时，将 `runner.resume_dir` 指向本地 `checkpoints/global_step_N` 目录，保留其 `actor/` 和完整的 DCP 检查点。用默认训练 YAML 做独立评估时，可加载本地权重。

```bash
CHECKPOINT='/absolute/path/to/full_weights.pt' GPU=0 \
  ./run eval example.yaml \
  'rollout.model=${actor.model}' 'runner.ckpt_path=${oc.env:CHECKPOINT}'
```

## 真机策略部署

使用 `deploy` 运行单机 Franka 策略。单机 Franka 的配置模板名为 `realworld_eval`。部署方可将模板保存为可编辑的 YAML，填入机器人地址、任务与设备参数，并把 `rollout.model.model_path` 指向受支持的本地模型或部署权重目录。保留 YAML 中的 `cluster.component_placement` 并使用该配置的设备分配。

```bash
./run deploy realworld_eval --cfg job
app=$(./run --runtime-dir)
cp "$app/app/evaluations/realworld/realworld_eval.yaml" ./my_realworld.yaml
```

在 `my_realworld.yaml` 填好设备地址、相机和模型路径后，先预览，再执行。

```bash
./run deploy ./my_realworld.yaml --cfg job
./run deploy ./my_realworld.yaml
```

配置预览只组合 YAML。最后一条命令会在配置指向的真机上运行策略。默认离线环境只覆盖上述仿真任务。真机控制所需的 SDK、控制主机、相机和模型文件须另行准备，并按设备与任务验收。交付版验收覆盖单节点真机配置预检。启动器使用单机本地 Ray。双节点 YAML 仍可用 `--cfg job` 预览，直接执行时会在初始化设备前提示需要多个节点。

## 可选的 RStar2 Code Judge

RStar2 使用独立部署的 Code Judge 服务。客户端沿用配置里的服务地址和 `/run/long-batch` 协议。每次提交会发送固定的 Python 加载语句、模型生成的代码和标准输入。沙箱中的原生扩展负责执行该提交。

在能访问解包文件与 Code Judge 沙箱文件系统的主机上执行以下步骤。将 `judge_python` 改为**实际执行提交**的 Linux x86-64 CPython 3.11 解释器路径。若服务启动独立容器，将扩展复制进每个沙箱镜像里的对应目录。

```bash
app=$(./run --runtime-dir)
judge_python=/path/to/code-judge-sandbox/bin/python3.11
judge_site=$("$judge_python" -IB -c 'import sysconfig
print(sysconfig.get_path("platlib"))')
extension=( "$app"/app/*_rstar2_sandbox*.so )
install -m 0644 "${extension[0]}" "$judge_site/"
module=${extension[0]##*/}
"$judge_python" -IB -c 'import importlib, sys
assert callable(importlib.import_module(sys.argv[1]).run)' "${module%%.*}"
```

Code Judge 应在隔离环境中运行，每个提交启动一个独立的 Python 进程。交付包提供沙箱扩展。完整 Code Judge 服务需要独立部署，本文中的验证只覆盖本地固定提交和协议对照。
