# TEM 指定颗粒追踪：研究与架构决策

检索日期：2026-09-28。先完成资料比较和视频检查，再实现；以下“推荐”是针对本视频的可验证工程选择，不是未经测量的准确率排名。

## PART 1 — 旧系统重建

根据用户提供的项目说明，历史路线是自定义 detector，后来使用单类 YOLOv8n；关联为 6 维 Kalman `[x,y,w,h,vx,vy]` 与 Hungarian，代价 `0.7*(1-IoU)+0.3*abs(A-A0)/A0`，阈值 0.7，max_age 15，轨迹历史 50。没有证据证明旧系统使用过 ByteTrack 或 `model.track()`。

检查了本工作目录以及说明中的 `E:/videoProjectLhy/videoProjectLhy`、桌面 `data`、`track`；后三者均不存在，本目录无旧源码、标签或权重。因此只能重建文档 baseline，不能声称复现了原模型性能。原始视频与参考图只读，不改动。YOLO 训练/推理入口保留；没有 TEM 权重时不使用 COCO 检测器冒充粒子检测器。

## PART 2 — 问题特征与实验约束

用户已明确：**只追踪手动指定的颗粒**。本次确认首帧编号 1、2、3、4、6、7；5 需重新定位，8 不确定，先排除。输入为 TEM 视频和人工首帧圆形 ROI，输出绿色圆圈、黄色中心轨迹、持久 ID、逐帧坐标、丢失状态及统计。属于**人工初始化的多条 single-particle/ROI tracking**；不是自动发现所有颗粒，不是原子柱追踪，也没有要求推断合并/分裂事件。

实际解码：554×554，26 帧，容器 1 fps、26 秒播放时长。画面时间标注从 0.0 s 到 5.0 s，说明播放帧率不能直接作为采集帧率。可见标尺 10 nm，首帧白条约 175 px；`10/175≈0.0571 nm/px` 仅为待确认的图像估计，不自动启用。用户要求暂不标定，后续可更改，因此配置中采集帧率与 nm/px 均保留 null。

颗粒多为几十像素宽的暗对比斑块，集中在左侧，具有可见形变、密度变化、邻近结构和边界截断；本文件仅 26 帧，不能按“50–200 帧人工子集”机械抽取。应检查全部 26 帧及多颗粒标注。亚像素数值插值可实现，但不等于已证明亚像素物理精度。

背景区域配准预检：phase correlation 相邻帧响应中位数约 0.550；ECC 相关系数中位数约 0.700；背景 ORB 合格匹配数中位数 0。phase 累计位移约 (-30.93,-7.65) px。第 10→11 帧相位法与 ECC 分歧明显。应使用无颗粒背景 ROI、质量门限及失败标志；不能以全颗粒平均位移直接宣称载物台漂移。

## PART 3 — 成熟方法调查

### 通用 MOT

- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/)：当前在线文档已包含 ByteTrack、BoT-SORT、OC-SORT、Deep OC-SORT 等多个后端。实际复现必须显式指定后端和包版本，不能依赖变化中的默认值。
- [ByteTrack](https://github.com/FoundationVision/ByteTrack)：两阶段关联可挽救低检测分数框；它解决的是漏检关联问题，前提是本域 detector 能发现颗粒。
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)：增加相机运动补偿和可选 ReID；TEM 相似灰度颗粒不保证有可区分的人体式外观特征。
- [SORT](https://github.com/abewley/sort)、[DeepSORT](https://github.com/nwojke/deep_sort)、[StrongSORT](https://github.com/dyhBUPT/StrongSORT)：分别代表轻量运动基线及外观增强路线。原 DeepSORT 的行人特征权重不能直接当 TEM 身份证据。
- [OC-SORT](https://github.com/noahcao/OC_SORT) 与 [Deep OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT)：观测中心运动修正适合非线性轨迹；后者再加入外观。不能把行人 benchmark 分数当本视频实测结果。
- [Norfair](https://github.com/tryolabs/norfair)：可自定义点距离与 Kalman 关联，对中心坐标比只依赖 bbox 更灵活；仍需可靠候选检测。
- [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw)、[FairMOT](https://github.com/ifzhang/FairMOT)、[CenterTrack](https://github.com/xingyizhou/CenterTrack)、[MOTR](https://github.com/megvii-research/MOTR)、[TrackFormer](https://github.com/timmeinhardt/trackformer)：分别利用框回归、联合检测/ReID、中心位移和时序 transformer 查询。迁移到 TEM 需要本域数据与训练，本次人工选点并无优势证据。

### Scientific particle tracking 与 microscopy

- [Trackpy 0.7 API](https://soft-matter.github.io/trackpy/v0.7/api.html) 把定位、亚像素细化与 linking 分离；适合输入灰度斑点中心，支持 memory、search_range、predictor 和轨迹分析。`search_range` 应覆盖相邻帧最大可信位移，并尽量小于邻粒间距；增大 memory 并不能解决真正身份歧义。[Linking 说明](https://soft-matter.github.io/trackpy/v0.7/tutorial/subnets.html)
- [TrackMate](https://imagej.net/plugins/trackmate/) 是成熟 Fiji 交互检查选择；适合科学人员修订 spot/track，增加 Fiji 依赖后可作为独立复核工具。
- [DeepTrack2](https://github.com/DeepTrackAI/DeepTrack2) 提供成像仿真、数据流水线和 LodeSTAR 等例子；其 2.x 与旧 TensorFlow 教程存在代际差异。要用于 TEM，必须验证模拟的电子成像对比、噪声和颗粒形态与真实数据匹配，不能把光学球形粒子仿真直接视作 TEM ground truth。LodeSTAR 值得在传统局部定位失败后测试；MAGIK 图关联需额外训练和验证。
- [AtomAI](https://github.com/pycroscopy/atomai) 对原子分辨结构分析更相关；本视频颗粒级形态不需要先引入原子柱模型。[DeepCell](https://deepcell.readthedocs.io/en/master/) 与 [ilastik](https://www.ilastik.org/documentation/tracking/tracking) 有细胞/对象工作流，但必须重估 TEM 域差异。[Cell Tracking Challenge](https://celltrackingchallenge.net/) 可借鉴标注和评估，而非直接转用其排名。

### 分割、小目标与点跟踪

- [SAM 2](https://github.com/facebookresearch/sam2) 支持视频提示分割；[SAM 3](https://github.com/facebookresearch/sam3) 提供更广的提示能力。二者可用于 mask 和人工纠正，但通用预训练对暗弱 TEM 颗粒的可靠性尚未实测。
- [Cellpose](https://github.com/MouseLand/cellpose)、[StarDist](https://github.com/stardist/stardist)、[Mask R-CNN / Detectron2](https://github.com/facebookresearch/detectron2)、[YOLO segmentation](https://docs.ultralytics.com/tasks/segment/) 可表达形状与面积；星凸形状假设、细胞先验、粘连和低对比都可能限制 TEM 应用。圆圈 ROI 面积不可当作颗粒 mask 面积。
- [SAHI](https://github.com/Small-Object-Detection/SAHI) 用分块减少小目标缩放损失；当前 554 px、几十 px 颗粒无需优先切块。超分辨生成的纹理不应当作原始科学证据。
- [OpenCV LK/Farneback/ECC](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)：LK 可对手动点做局部位移估计，但颗粒内部低纹理和变化的干涉/噪声会漂移。Farneback 输出稠密流，不自带粒子身份。
- [RAFT](https://github.com/princeton-vl/RAFT)、[FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official) 提供学习光流；不能自行处理中心定义、出生/消失或科研置信度。[CoTracker3](https://github.com/facebookresearch/co-tracker) 与 [TAPIR](https://github.com/google-deepmind/tapnet) 更直接匹配手动点查询，可作为高运动难例的后续挑战者，需测 TEM 域偏差与点是否还位于粒子中心。
- [Stone Soup JPDA](https://stonesoup.readthedocs.io/en/latest/auto_tutorials/08_JPDATutorial.html) 与 [btrack](https://github.com/quantumjot/btrack) 为概率/图关联复用基础。Kalman、EKF、UKF、particle filter、MHT、JPDA 或 min-cost flow 的复杂度只有在可标注的密集歧义数据上才值得增加；它们不能恢复图像中已经不可辨认的物理身份。

## PART 4 — 候选路线比较

下表为基于数据的适配判断，并非实验准确率。完整维度见 `method_comparison.csv`。

| 路线 | 本数据可用前提 | 优势 | 主要失败模式 | 本次位置 |
|---|---|---|---|---|
| YOLO + ByteTrack/BoT-SORT | TEM 标签、权重 | 检测器可扩展；漏检恢复 | 域外检测；ReID 同质；框关联漂移 | 备选 |
| YOLO + 旧 Kalman/Hungarian | 旧权重或重新训练 | 延续历史、可解释 | 小框 IoU 敏感；形变；常速失效 | 保留接口，完整模型未实测 |
| YOLO + OC-SORT | 本域检测器 | 对非线性运动的观测修正 | 不能弥补错误检测和不可辨认目标 | 备选 |
| YOLO + Norfair | 候选中心可靠 | 自定义点距离 | 门限和漏检敏感 | 备选 |
| Particle localization + Trackpy | 暗斑可分离且尺寸合理 | 成熟科学 linking；亚像素细化 | 粘连、尺度变化、密集交换 | 第二选择，实际对照 |
| Segmentation + mask association | mask 可重复 | 面积/形状有科学定义 | 灰度阈值不稳定；粘连事件身份 | 未来形态实验 |
| Registration + localization/tracking | 固定背景 ROI | 分离共同位移 | 背景不固定或文字参与配准 | 作为独立可开关模块 |
| Detection once + optical flow | 局部纹理稳定 | 手动初始化即可；CPU 快 | 累积漂移、亮度恒定失效 | 实际 LK 对照 |
| Manual ROI + local NCC + center refinement | 目标仍可辨认，位移小于门限 | 无训练，保留选定 ID，质量可审查 | 相似邻粒、模板污染、强形变 | 本次首选实现与验证 |
| CoTracker/TAPIR | 预训练权重及足够计算 | 直接查询点时序跟踪 | TEM 域偏差；点不等于质心 | 后续深度模型挑战者 |

## PART 5 — 推荐架构

先实现：人工点选中心/半径 → 温和空间去噪 → 局部归一化互相关 NCC 搜索（OpenCV）→ 亚像素峰值及局部中心细化 → 质量门限、冲突拒绝、短暂丢失状态 → 持久 ID → 绿色 ROI 与黄色轨迹。不会对时间轨迹平滑造出不存在的运动。LK 提供对照和质量诊断。Trackpy 与重建 Kalman 使用同一科学定位器做关联对照。

背景配准使用独立 ROI；比较 phase、ECC、LK 与 ORB 后，优先 ECC/phase 质量检查的平移模型。保留原始坐标和背景相对坐标；配准不可靠时标记失效。主要视频显示原始图像坐标，画圈必须贴着原颗粒，不能在原图上直接画减漂移后的点。

## PART 6 — 为什么适合这次 TEM 实验

选定颗粒数量少、首帧位置已知，无需先训练全画面 detector；局部范围搜索比在全图“找相同黑斑”更约束身份。图像中目标直径几十像素，局部形状与暗对比是可用证据。OpenCV 与 Trackpy 提供成熟可解释组件，CPU 即可复现。失效时输出缺测和复核点比不间断的假轨迹更有意义。

## PART 7 — 实现计划

模块划分为 video/config、preprocessing/localization、registration、tracking、analysis、visualization。提供浏览器本地选点工具、CLI、config、运行记录、CSV、MP4、每粒轨迹图、对照实验、数据完整性验证与可选 YOLO 训练入口。原文件保持只读，每次结果写入新目录。输入和参数保留可追溯摘要。

## PART 8 — 评估计划

1. 为已选颗粒做原图目视标注子集；任何助手标注都标明“初步目视参考、非专家 GT”，不混作正式 ground truth。
2. 本数据实际对照：所选方法、LK、Trackpy、旧 Kalman 关联；YOLO 缺权重记为 not_run，不伪造分数。
3. 控制变量：raw/blur，with/without registration，search_range 扫描，legacy IoU / IoU+area / IoU+distance / motion；保持同一输入及种子。
4. 核验每帧 ID 唯一、丢失无测量坐标、轨迹不跨缺测连线、回读视频帧数、坐标与覆盖层一致、原始文件 hash 未改变。合成测试覆盖平移、遮挡消失、边界和漂移符号。
5. 真实 GT 可用时报告中心误差、指定身份成功率、漏跟踪及错误身份；HOTA/IDF1/MOTA 需完整身份标注与合适评估设置。没有 GT 不用“轨迹更顺滑”宣称更准确。

## PART 9 — 风险与未解决事项

- 第 5、8 号首帧定义待用户重新选定。本次不纳入。
- 对比度变化会移动强度质心；ROI 不是分割 mask。不能从固定半径宣称面积变化。
- 临近颗粒接触、遮挡、融并后物理身份可能不可辨认；算法应标缺测/不确定，不能强行延续。
- 背景相对运动不一定等于绝对载物台运动；真实整体协同运动也可能被配准消除。
- 采集时间与像素物理标定待确认。用户明确要求记录、以后可补；配置和 CSV 保留对应字段。暂不输出扩散系数，不把 26 帧压缩视频当原始科研数据。
- 推荐不是“所有 TEM 最佳”。外部权重、更多 TEM 视频、专家标注和独立测试集仍是正式科研结论的前提。

## PART 10 — 环境与来源

正文每项均链接官方文档、作者仓库或 benchmark。额外参考：[Python releases](https://www.python.org/downloads/)、[PyTorch 官方安装](https://pytorch.org/get-started/locally/)、[PyTorch PyPI](https://pypi.org/project/torch/)、[Ultralytics PyPI](https://pypi.org/project/ultralytics/)、[Trackpy PyPI](https://pypi.org/project/trackpy/)、[OpenCV phase correlation](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html)、[NCC template matching](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html)。

实际使用独立 Python 3.12 环境，版本锁定来自运行成功后的 `pip freeze`；机器有 RTX 4060 Laptop GPU，但主流程仅需 CPU。查询“最新版”与选择“本项目可复现版本”分开。PyTorch 官网安装选择器的抓取内容与包注册表可能不同，以锁文件和硬件实际验证为准；未安装/未测试的 GPU 路线明确标记。


## 补充：精细定位与预处理边界

[径向对称中心原始论文](https://www.nature.com/articles/nmeth.2071) 提供基于径向对称的解析定位；[TrackMate 检测器](https://imagej.net/plugins/trackmate/detectors/) 提供 LoG/DoG 和其他 spot/mask 方案。高斯拟合、LoG/DoG、径向对称都值得在近对称斑点上测试，但本视频颗粒有显著非对称、内部对比变化，不能仅因支持亚像素就默认更准确。本次选择可审查的局部相关峰值与受限中心细化，仍不声称真实亚像素精度。

高斯与 raw 已完成受控对照。CLAHE 已提供开关，但未凭主观清晰度启用；median/NLM、背景扣除、带通/Fourier 和直方图匹配未加入默认流程。它们可能改变局部强度中心、削弱邻粒边界或去除真实结构，应分别增加控制实验后再采用。当前比较表中的 CPU/实时性为定性判断，只有 results 中的运行记录属于本机实测。

## 补充：2026-09-28 注册表快照

官方 PyPI JSON 查询保存在 package_registry_snapshot.json。查询结果：torch 2.14.0，ultralytics 8.4.164，opencv-python-headless 5.0.0.93，scipy 1.18.1，trackpy 0.7，deeptrack 2.0.2，norfair 2.3.0。只把实际成功运行的主环境依赖写进 requirements-lock.txt。GPU 为 RTX 4060 Laptop GPU，驱动检测为 610.88；没有执行 torch/CUDA 兼容性试验，故不声称某 CUDA 安装组合已验证。

最终排序是针对实施路径：主选人工 ROI + OpenCV 局部追踪；第二选择 Trackpy；对照基线为重建 Kalman/Hungarian。选择依据是本次手动指定 ID、无训练集、图像特征及可复核性，不是未经证明的精度冠军。
