# 可训练 TEM 颗粒追踪：2026-09-29 架构更新

## 需求分类与旧工作

用户已确认：“可标注、可训练并迁移到新视频；模型先识别颗粒，我选择需要追踪的目标”。当前是**监督式颗粒检测 + 用户选择 + 多条指定身份的 detection/association**。训练标签须完整标注选定帧，输出轨迹仅限用户选择目标。

没有环境控制动作、奖励函数或策略学习要求，故不采用强化学习。人工纠错后重训是监督微调；推荐下一批标注帧是不确定性/多样性采样。

第一版已检查旧项目路径、视频与候选文献，旧源码和 TEM 权重未找到，重建 baseline 不代表恢复了原模型性能。详见 [旧研究](research_report.md)。本次保留模板、光流、传统定位、背景配准、统计与可视化，新增学习数据管理、模型训练与版本，以及学习检测的关联接口。

## 所有候选路线按新目标复评

下表是工程适配判断，不是已完成的 TEM 准确率排名。完整来源与旧实测见 [研究报告](research_report.md)、[候选表](method_comparison.csv)、[旧实验记录](experiment_results.md)。

| 候选 | 本次判断 |
| --- | --- |
| NCC 模板、LK 光流 | 保留基线。利用指定起点，但不能承担跨实验外观学习；形变、低对比、累积漂移会失效。 |
| Trackpy 定位/关联；TrackMate、u-track | 科学颗粒生态成熟。当前不规则背景下传统定位需调参；中心关联保留，与相同学习检测比较，稀疏或跳跃运动可能优于 IoU。 |
| YOLO26n/YOLO11n/YOLOv8n + ByteTrack | 框标注与迁移微调成本适合首轮建设，成熟训练与关联可复用。采用 YOLO26n 为工程起点，未宣称在本 TEM 视频胜过 YOLOv8n。 |
| SORT、历史 Kalman/Hungarian、Norfair | 轻量、检测与运动分离。重建 6D Kalman/Hungarian 接收同一学习候选作对照；中心路线由 Trackpy 实现。 |
| OC-SORT、BoT-SORT | 复杂运动或相机运动时值得比较。当前独立背景配准便于核查，不直接假定通用 ReID 可区分 TEM 颗粒。 |
| DeepSORT、StrongSORT、Deep OC-SORT、FairMOT | 需要有区分度的身份外观；相似灰度颗粒暂无通用外观权重有效的证据。需本域身份标签和独立收益证据。 |
| Tracktor、CenterTrack、MOTR、TrackFormer | 回归、中心位移、时序查询有相应数据需求；当前样本不足以证明复杂时序模型收益，暂不优先。 |
| Cellpose、StarDist、U-Net、SAM 系列、分割追踪 | 真正面积、边界或接触颗粒拆分成为目标时应重评。当前中心轨迹需求下框标注成本较低；框面积不冒充掩膜面积。 |
| LapTrack、btrack、图/概率关联 | 密集身份歧义时值得研究，但不能用复杂算法解决本身不可辨认的身份。 |
| LoG/DoG、Gaussian、径向对称定位、Atomap 等 | 适合满足成像假设的定位细化或原子柱问题。当前未要求原子级精度，也无精度标定，不能把浮点框中心称为亚像素物理测量。 |
| 强化学习 | 当前是标签监督的识别任务，不建立与目标无关的动作/奖励训练环境。 |

## 实现选择和原因

PyTorch 提供训练及 GPU/CPU 运行，Ultralytics 提供单类 `particle` 检测训练、推理、验证。默认 YOLO26n 从官方通用权重初始化，TEM 能力由确认的 TEM 标签学习；n 模型便于在本机 8 GB GPU 迭代。配置允许切换 YOLO26s、YOLO11n、YOLOv8n 作控制比较。[YOLO26 文档](https://docs.ultralytics.com/models/yolo26/)、[训练文档](https://docs.ultralytics.com/modes/train/)。

在原灰度图标注，输入复制为三通道。默认关闭 mosaic、mixup、色相和饱和度变化，保留适度旋转、翻转、亮度等扰动；所有设置保存，需以固定验证集判断效果。增强不覆盖原图。

ByteTrack 采用锁定版本的官方实现，利用高/低分两阶段关联；它不能找回根本没有生成候选的颗粒。保存原检测索引，输出原图检测框中心，Kalman 坐标仅作关联诊断。用户 ID 与后端 ID 分开，丢失后不自动换到邻近新身份。[ByteTrack 作者仓库](https://github.com/FoundationVision/ByteTrack)、[接口文档](https://docs.ultralytics.com/reference/trackers/byte_tracker/)。

新入场的高分暂定轨迹允许在第一帧被人工选中，后续仍需真实检测；不因此输出无观测预测点。当前颗粒外观相似，没有引入人体 ReID 网络。Trackpy 直接使用同一检测的中心坐标，便于判断是检测失效还是关联失效。[Trackpy link_df](https://soft-matter.github.io/trackpy/v0.7/generated/trackpy.link_df.html)。

| 参数 | 默认值与意义 |
| --- | --- |
| inference confidence / track_low_thresh | 0.10 / 0.10；保留低分候选，检测门限不能更高。 |
| track_high_thresh | 0.40；第一阶段候选门限。 |
| new_track_thresh | 0.50；创建新轨迹门限，界面可调，下调时同步限制 high。 |
| track_buffer | 3 帧；后端保留失联身份的时间，不是填补预测坐标。 |
| match_thresh | 0.80；匹配代价门限，越大允许越松。 |
| fuse_score | false；避免未校准低分再次压低几何相似度，须在本域验证。 |
| max_link_step_px | 50 × √间隔；额外位移检查，不是测定的物理速度上限。 |
| Trackpy search_range / memory | 28 px / 2 帧；需和实际最大位移、漏检长度比较。 |

Ultralytics 8.4.165 的 ByteTrack 构造函数只接收 args，buffer 直接以帧计，已经测试核实。以上是可复现起始设置，尚无真实 TEM GT 证明最优。

## 数据和模型闭环

标注具有草稿/确认状态及修改历史，模型建议不自动当真值。按实验划分并核查完全重复图像；固定版本保存 PNG、YOLO 标签、原始视频哈希、框坐标、标注 revision。训练前检查文件哈希；继续微调还检查父模型已见数据，防止验证泄漏。

每个模型保存权重哈希、数据来源、父模型、实际 GPU/CUDA、配置、验证结果。通用初始化和合成检查权重不出现在 TEM 模型列表。训练运行在独立进程，界面仍可标注、读取进度和停止任务。

背景 phase/ECC 配准与质量检查沿用旧工作，失败后的累计相对坐标缺测。原始坐标保留，默认不强行用于关联。没有确认的物理标定时不输出伪精确物理速度。

## 比较和验证计划

`compare_learned.py` 在同一组学习检测上实际运行 ByteTrack、Trackpy、历史 6D Kalman/Hungarian。原来的漂移、预处理、模板、光流对照保留。检测用 precision/recall/mAP；追踪需补独立人工身份点、中心误差、错 ID、漏测和轨迹碎片化；完整 MOT 分数需要对应完整身份 GT 与评估协议。

验证集参与模型选择；迁移结论应来自未用于调参的新实验。真实 TEM 标签未到位，不能用合成成绩或旧模板观测率填充新模型效果。跨视频倍率、焦距、亮度和噪声变化会引起域偏移；严重重叠、合并/分裂与不可辨认身份需人工复核。

实际环境以运行检查为准，安装来源为 [PyTorch 官方入口](https://pytorch.org/get-started/locally/)；验证证据见 [learning_validation.md](learning_validation.md)。
