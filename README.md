# RLinf 编译交付工具

用 Cython 和 GCC 编译上游 RLinf 的业务模块，生成可以单独交给使用者的 Linux x86-64 代码包。训练、评估和单机真机策略部署共用一个可执行文件，运行者通过第一个参数选择任务入口，通过第二个参数选择 YAML 或内置配置名。

v0.2 默认把可迁移的任务运行时嵌入同一个 ELF。目标机离线安装，可以在没有源码、系统 Python 或模型下载缓存的环境里运行随包的 FrankaSim PPO/MLP 和 ManiSkill PushCube PPO/MLP。业务模块和 RStar2 沙箱扩展使用 Cython 和 GCC 编译，外层 ELF 启动器使用 GCC 编译。

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
├── run
├── install.sh
├── example.yaml
└── README.md
```

`run` 包含原生模块、配置组、独立 CPython 和任务依赖。`install.sh` 将环境解包到交付目录，校验成功后切换本地 `.venv`。`example.yaml` 默认取自 FrankaSim PPO/MLP。构建器会在默认示例中说明如何预览配置，并将默认项目日志名称设为 `training`。若上游版本缺少 FrankaSim 配置，构建器会选用具身入口的默认 ManiSkill 配置或该目录的首份 YAML。目录内的其他注册配置同样可以按名使用。

对外分发时，只复制上面四个文件。默认文件名、安装提示、示例配置、帮助和客户说明使用中性名称。编译产物内部保留上游 Python 包名、注册配置、许可证和署名，检查内部文件仍能识别上游项目。已有旧版交付时，可在新目录安装 v0.2，再用 `runner.resume_dir` 指向原检查点。发布方仓库跟踪构建工具、回归检查和说明。上游检出、生成的交付包、私有报告和本机实验记录均由发布方单独保管。

可以将另一份完整配置作为第二个位置参数，构建器会将其原样发布为 `example.yaml`。输出目录必须是新目录。如果根目录中有多个上游检出，可用 `--source` 指定。

```bash
./compile.sh release-sft RLinf/examples/sft/config/libero_sft_openpi.yaml --source RLinf
```

构建器优先复用项目根目录的 `runtime`。首次在新项目构建会用仓库内的锁定配方准备运行时和 PhysX GPU 库，下载发生在发布方机器上。构建机需要 x86-64 Linux、Python 3.10 或更高版本、Git、Bash、GCC/G++、GNU 工具、curl、dpkg-deb，以及 glibc 和 zlib 的静态开发库。也可以明确指定已准备好的独立 CPython 3.11 运行时和专属资源目录，并把审计报告留在交付目录之外。

```bash
./compile.sh release-offline --source RLinf --offline-runtime ./runtime \
  --offline-home ./task-home --audit-report ./release-audit.json
```

资源目录中只读取固定版本的 `.sapien/physx/.../libPhysXGpu_64.so`。如已保存该库或官方 `linux-so.zip`，可使用 `--physx-asset`。构建工具也支持通过 `RLINF_CPYTHON_ARCHIVE`、`RLINF_WHEELHOUSE` 和 `RLINF_{VULKAN,EGL,GLVND,OPENGL}_DEB` 提供本地输入。需要在目标机按上游模型、模拟器选项联网安装依赖时，用 `./compile.sh release-online --online-installer` 构建对应制品。[联网安装版说明](delivery/RELEASE.md)

如需附带已准备好的模型权重或数据，构建时加 `--offline-assets ./task-assets`。该目录随 ELF 安装到当前离线环境的 `assets/`，YAML 可使用 `${oc.env:TRAIN_OFFLINE_ASSETS}/model.safetensors`。允许 safetensors、Parquet、JSON/JSONL、YAML、UTF-8 文本/CSV/TSV 和 PNG/JPEG/WebP。准备专用数据目录，移出链接、隐藏项、凭证、脚本和压缩包。大型文本数据请用 JSONL。此选项只提供本地文件路径，任务 YAML 仍需指定模型与数据。

## 自动保护当前分支的私有改动

默认执行 `./compile.sh` 时，发布方同时得到 `release.protection/priorities.md` 和 `release.protection/report.json`。只将四文件的 `release/` 交给客户。每次构建都会将当次对照中被判为高优先级的整个业务模块交给强化编译器，再以 AES-256-GCM 封装。其余模块沿用普通原生编译。配置和训练入口一并保留，成品审计对已验证解密的模块执行同一套源码检查。

发布方的 Python 需要 `cryptography`（AESGCM），GCC 需要 OpenSSL EVP 头文件及静态 `libcrypto.a`。本项目在 Ubuntu 22.04 的 `python3-cryptography`、`libssl-dev` 上完成验证。目标机使用交付的静态启动器。若目标机通过 `OPENSSL_CONF` 限制允许使用的算法，需要允许 AES-256-GCM。该算法被策略禁用时，首次解包也可能报告模块认证失败。

每个构建使用独立的 256 位密钥，每个选中模块使用独立的 96 位随机 nonce，认证标签同时绑定模块路径。离线自动运行所需的密钥随启动器交付。首次解包把通过认证的原生扩展写入本地 `.train-app/`，合法客户可以读取扩展。加固精简了合成 Python 代码对象中的非参数局部变量名和表达式位置，保留函数调用签名。

### v0.2 验证记录

- 本项目的固定检出中，915 个业务模块自动选中 17 个。断网完成离线版、联网安装变体和零选中对照的构建。原离线验收从四个文件重新安装，在屏蔽源码和旧环境的条件下迁移到包含中文和空格的路径。FrankaSim PPO/MLP 实际完成一次全局训练更新、32 条评估轨迹及检查点保存。ManiSkill PushCube PPO/MLP 的训练、评估和保存已在先前 v0.2 离线验收中跑通。
- 最终命令表包含 25 条上游脚本路径和单独的 `deploy` 入口。用同一版本编译的启动器枚举包内 38 份评估主配置及 1 份具身目录的双臂评估配置，39/39 均选中预期目录并完成 Hydra 组合。其中 5 份真机配置也用 `deploy` 全部组合通过。测试包括 RTC 配置、日志路径覆盖，以及实际执行双节点配置时在设备初始化前退出。真机部分仅做配置和启动边界验证。
- 正式离线 `run` 在含中文和空格的新目录中，由四个文件从头安装。复验期间无外网，源码目录不可读。FrankaSim PPO/MLP 在物理 GPU 6 完成一次更新，保存的 16 个优化器步数张量均为 800。随后独立运行 `./run eval`，使用同一份导出权重，并通过 `rollout.model=${actor.model}` 指定完整策略模型，完成 32 条评估轨迹。
- 对加固包的独立盲测中，DAgger 筛选、Pico 控制及轨迹累积三个目标均无法用 `inspect.getsource` 读取 Python 原文，独立重写分别通过预留的 5、4、3 组行为用例。盲测使用已公开的历史源码、原生扩展和运行时观测。结果表明这三个目标的受测行为仍可复现。样本只覆盖上述用例，无法测定其余模块的复刻难度，也无法据此归因比较两版用时。

在项目根目录可以分别执行以下命令。每条命令中的报告目录须为新目录。

```bash
# 仅分析当前工作树，包含尚未提交且会进入离线发布的修改
./compile.sh --analyze-only --protection-report ./working-tree-review

# 复查已有四文件离线包，包括其 ELF、公开配置、离线环境和附带权重
./compile.sh --analyze-only --inspect-release release --protection-report ./release-review

# 在本机单独的公开 Git 镜像与分支之间找共同祖先，再构建完整离线包
./compile.sh release-next --source RLinf --public-repo ./public-rlinf.git
```

本项目默认从本地 Git 对象读取已核验的官方历史提交 `184ba07f0a5b3a09a4696a4e3c8aa4b3c720fa86`。提供单独的 `--public-repo DIR` 时，新增、修改和删除均相对于源仓库 HEAD 与公开镜像 HEAD 的唯一最近共同祖先计算。判断代码是否已经公开时，还会检索公开镜像 HEAD，在两个快照中跨路径查找字节相同的文件和语法树相同的完整定义。例如，分叉后双方都采用了某个函数，报告仍记录分支上的新增，同时下调该公开函数的保护优先级。`report.json` 保留命中的提交和路径，函数还记录限定名与行号。也可以用 `--public-repo DIR --public-ref SHA` 固定单个公开快照。本地 Git 可使用普通检出或裸镜像。分析在本地完成，若对象缺失或浅历史不足，需在发布方备好历史或指定已有的准确提交。自行指定的镜像和提交由发布方确认来源。[本项目的公开历史基线](https://github.com/RLinf/RLinf/commit/184ba07f0a5b3a09a4696a4e3c8aa4b3c720fa86)

`priorities.md` 按文件列出优先保护和人工复核项，`report.json` 给出限定函数名、改动行号、判断依据、删除项及公开实现复用情况。跨步状态更新、筛选决策、训练逻辑等是高优先候选。纯文本改动、简单映射及公开定义复用单列。可编辑 YAML、运行数据、可提取的模型资产与上游 Shell 入口差异需要按实际交付方式复核。报告仅记录名称、位置、判断依据和摘要，不复制源码、权重或 YAML 正文。离线环境中的常规文件接受完整性和源码泄漏检查。第三方依赖的私有改动需要相应依赖的独立基线。优先级是可解释的代码筛选规则，业务价值与实际逆向难度仍需人工和盲测确认。

普通离线构建将改写前及送入 Cython 前的源码摘要、ELF 成员、配置和离线资产绑定到发布方报告。构建前预览仅反映当前工作树及本机 Python 的暂存重放。跨 Python 版本的文本生成可能不同。已有发行包复查会核验实际成员、校验和与公开资源，缺少原始构建清单时，私有源码归属依据提供的工作树推断。保留正式构建同时生成的报告可供后续追溯。传入 `--skip-protection-report` 仅省略发布方报告文件，自动保护与成品审计仍会运行。联网安装版也会在构建时自动选择并封装模块。

## 发布前的源码检查

维护发布工具时，可运行 `python3 -B delivery/check_source_audit.py` 执行源码泄漏回归。

构建器在发布目录生效前检查最终 ELF 和三份同级文件。它从成品解析内嵌归档、Cython 的压缩字符串池、原生扩展的调试节和可读文本，对照上游 `rlinf`、`examples`、`evaluations`、`toolkits` 的源码和内嵌程序。对于离线运行时，会检查归档目录、链接、依赖文本与嵌套 Python 包。命中受保护源码时构建失败。

带加固模块的离线包可用上节的 `--inspect-release` 重新验证归档与源码。另可用 `python3 -B delivery/source_audit.py release/run RLinf --sidecars release` 输出源码审计 JSON，其中 `RLinf` 是构建所用的检出目录。命中受保护源码会返回非零退出码。

旧版的 RStar2 客户端会在评测请求中发送 156 行 Python。新版将实现编译成独立沙箱扩展，客户端只发送固定的导入调用和提交数据。使用 RStar2 的运营方需要在现有隔离 Code Judge 的 CPython 3.11 环境安装该扩展。普通具身训练与评估无需此服务。[离线交付说明](delivery/OFFLINE.md)

交付目录仍提供可阅读的 YAML、说明和安装器。CPython、PyTorch、Ray、模拟器等第三方依赖包含自身的源码、头文件、许可证和文档。原生程序中也可见符号、常量和报错文字。这些内容及机器码均可由收件人检查。

## 使用

把整个 `release/` 复制到目标机器。在默认示例上，可用以下命令安装环境、查看组合后的配置并启动训练。

```bash
cd release
./install.sh --offline
./run embodiment example.yaml --cfg job
./run embodiment example.yaml
```

使用 VLA SFT 时，选择 `vla_sft` 并提供该任务的 YAML 或注册名。其他入口见 `./run --help`，也可使用兼容的原脚本名或路径。具身评估使用 `./run eval`，单机 Franka 策略的配置预检使用 `./run deploy realworld_eval --cfg job`。要运行注册配置对应的任务，需在交付前备齐依赖、权重、数据和模拟器资产，把 YAML 指向目标机上的本地路径。默认离线包覆盖上述两个从随机初始化开始的 MLP 任务。其他配置需要独立验收。[交付使用说明](delivery/OFFLINE.md)

自定义 Franka 任务时，可复制完整的 `realworld_eval.yaml` 再编辑。Hydra 要求 `hydra.searchpath` 出现在主配置中，将带有该字段的模板作为另一份 YAML 的 `defaults` 子项会导致组合失败。

部署单机 Franka 时，保留 `cluster.component_placement` 里的硬件节点组，并准备匹配的 OpenPI、Franka SDK、设备连接、权重和归一化统计。发布方如需离线交付这些依赖，可以在已通过离线安装校验的运行时上补齐任务库，再用 `--offline-runtime` 构建。SDK 的系统驱动仍需在目标机上准备。交付启动器固定使用本地 Ray。`realworld_pnp_eval` 等上游双节点配置可以用 `--cfg job` 预览。若直接执行，启动器会在创建 Ray 集群或初始化设备前报告拓扑不兼容。

评估器在主 YAML 没有声明 Hydra 搜索路径时补入随包具身配置组，使两份 RTC 评估模板也能按注册名组合。配置中已声明的搜索路径和命令行覆盖优先。由训练配置派生独立评估时，如模型完整定义在 `actor.model`（例如双臂 Franka 模板），需要在 YAML 填齐 `rollout.model`，或显式传入 `'rollout.model=${actor.model}'`。

扩散策略评估使用 `eval_diffusion`，从扩散任务的配置目录选择 YAML。以下命令只检查 Wan 配置组合，实际评估需另备模型、数据及对应依赖。

```bash
./run eval_diffusion wan22_ti2v_5b_nft_video_ocr --cfg job 'rollout.model=${actor.model}'
```

`toolkits/standalone_eval_scripts/` 中的独立工具各有自己的参数解析、依赖和配置方式，未注册到统一入口。使用这些工具时按相应源码目录里的使用说明执行。

## 单机资源起步配置

以下为状态输入、关闭视频和 rollout Torch 编译的单卡 PPO/MLP 配置。训练和评估环境数与实测长跑一致。显存、内存和磁盘容量是给首次部署预留的预算，尚未测得端到端的最低硬件配置。

| 任务 | 训练 / 评估环境数 | CPU | 主机内存 | GPU | 可用磁盘 |
| --- | ---: | ---: | ---: | --- | ---: |
| ManiSkill PushCube-v1 | 256 / 32 | 8 个逻辑 CPU | 24 GiB | 1 张 NVIDIA GPU，预留 8 GiB 显存 | 25 GiB |
| FrankaSim | 128 / 16 | 8 个逻辑 CPU | 32 GiB | 1 张 NVIDIA GPU，预留 16 GiB 显存 | 25 GiB |

PushCube 短测完成 3 步训练、评估与保存，采样峰值约为 9.9 GiB 主机进程内存（PSS）和 4.4 GiB 显存。FrankaSim 完成 1 步训练、评估与保存，采样峰值约为 18.1 GiB 主机进程内存和 11.3 GiB 显存。关闭视频时 MuJoCo 仍会分配 NVIDIA EGL 图形上下文。

从 Ubuntu 22.04 x86-64（glibc 2.35）和兼容 CUDA 12.8 的 NVIDIA 驱动环境起步。实测驱动为 570.172.08，GPU 为 A800-SXM4-80GB（计算能力 8.0）。PushCube 需要系统 NVIDIA Vulkan ICD。Ray 对象存储上限设为 4,000,000,000 字节时，预留至少 4 GiB 可用 `/dev/shm`，并提供可写的短路径临时目录。其他 GPU、驱动、视觉模型和 SFT 任务需要按配置重新测量。
