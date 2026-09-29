# TEM 指定颗粒追踪

> 这是第一版的历史说明。V2 的安装与启动请阅读项目根目录 README.md；传统方法在新版界面的“传统方法对照”入口继续使用。

在视频中点选颗粒，程序用绿色圆圈跟随它们，并用黄色线记录中心轨迹。支持持久 ID、逐帧检查、同一 ID 的中途校正点、MP4 和 CSV 导出。所有运算在本机完成，原始输入只读。

本次样例已处理 **1、2、3、4、6、7**。第 5 号需重新选中心，第 8 号尚未确认，两者没有纳入。采集帧率和 nm/px 均保留未标定，已写入 [实验记录](EXPERIMENT_NOTES.md)。

## Overview / 方法选择

这次任务属于“人工初始化的多条单颗粒/ROI 追踪”。主流程使用 OpenCV 的局部归一化互相关、亚像素峰值插值、受限暗强度中心细化及质量检查，无需训练检测器。它利用已知起点和短距离搜索约束身份；低对比、相似邻粒和形变仍可能导致失效。丢失时留空，可由人工校正恢复。

对照路线为 LK 光流、Trackpy 科学颗粒定位与关联、重建的 Kalman + Hungarian。旧 YOLO 模型未找到，保留可选训练/检测入口，但没有伪造它的实验成绩。现有早期参考点不足以证明某方法全面最优。

- [研究与架构选择](docs/research_report.md)：官方文档、作者仓库、旧方案重建及选择依据。
- [完整方法比较表](docs/method_comparison.csv)：24 个维度及来源，定性适配判断。
- [实测对照与限制](docs/experiment_results.md)：14 组已执行实验及未执行项。

## Environment Setup / 启动

当前电脑已经配置好环境。**双击 `start_tracker.cmd`**，浏览器会打开本地选点界面。关闭网页不终止后台服务；下次启动会复用同一项目的服务。界面默认端口 8765，冲突时自动尝试后续端口。

在另一台 Windows 电脑：安装 Python 3.12，解压整个项目，双击 `setup.cmd` 安装锁定依赖，再双击 `start_tracker.cmd`。首次安装需要网络。环境放在项目 `.venv`，不覆盖全局 Python。

也可在项目目录运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\serve.py
```

测试环境：Python 3.12.14、OpenCV 5.0.0、Trackpy 0.7、NumPy 2.5.3、SciPy 1.18.1、pandas 3.0.6。完整版本见 `requirements-lock.txt`。主流程使用 CPU，不需 CUDA。RTX 4060 Laptop GPU 已检测到，但 PyTorch/CUDA 路线未执行兼容性实验。

## 界面操作

1. 示例已填入 6 个起始点，点击“追踪所选颗粒”即可重跑。或选择文件并点击“打开新视频”，新视频会复制到 `data/imported/`。
2. 在画面点颗粒中心，调整“圆圈半径”。这是跟踪 ROI，建议包含目标和少量周围纹理，尽量不包含邻粒。编号按新增顺序产生，已有 ID 不重排。
3. 拖动时间条检查每帧；可关闭“显示轨迹”查看原始图像。帧号从 0 开始。
4. 若某帧开始偏离，在“点击画面的操作”中选“校正颗粒 #ID”，切到该帧，重新点击正确中心，再追踪。校正只影响该帧及其后，不会修复更早的帧；需要时在更早帧补点。
5. 缺测和人工校正前后不会直接连线，也不计算跨越该间隔的瞬时速度。不要为图上连续而给不可辨认目标指定身份。
6. 通过“导出选点”保留锚点 JSON。每次运行也自动保存选点和参数。结果链接可打开视频、CSV、待复核清单和轨迹图。

界面显示的是上次已完成运行的轨迹。改变选点后请重新追踪。运行结果写入新的 `results/manual_日期时间_编号/`，不会覆盖前次实验。

## Dataset / 数据

`data/raw/sample.mp4` 是所给原视频的副本，554×554，共 26 帧，容器 1 fps。画面自带实验时间 0.0–5.0 s，与播放时间不同。时间标记暗示的 5 fps、标尺估计的约 0.0571 nm/px 均未自动采用。将来确认后在界面“追踪参数与标定”中输入，或改配置重新运行。

`data/annotations/visual_reference.csv` 包含助手目视估计的早期参考点，仅用于初步定位检查。它不是专家 ground truth，且非盲评；准确范围和来源见同目录 README。不要用同一视频的相邻帧随机切分来声称独立泛化验证。

## Detector / 定位器

主方法从人工 ROI 开始，不做全图检测。高斯空间去噪 sigma=2 px，NCC 使用圆形模板掩膜，峰值插值后做最大 6 px 的局部中心细化。NCC 为相似性分数，不是正确率/概率。

Trackpy 对照使用暗斑反相、直径 61、minmass 13000、separation 39、percentile 65。该样例上/下边缘 40/45 px 含文字或标尺，检测器通过 `exclude_top_px`/`exclude_bottom_px` 排除；换数据时重新设置，主手动 ROI 方法不使用这些检测排除带。

## Tracker / 关联与质量门限

默认 `template`：搜索半径 28 px，memory=2，NCC≥0.55，峰值差≥0.025，反向误差≤12 px，最小对比度 4。搜索位置以原始坐标中的上一测量为基础，可加入可信背景步进；接受新测量后更新局部模板。质量不足输出 `uncertain`，连续缺测超过 memory 后 `lost`，此时需要手动锚点恢复。memory 期间恢复仍可能认错，必须检查身份。

`lk` 使用 OpenCV 金字塔光流和前后向检验。`trackpy` 使用科学定位后按中心距离关联；该对照入口每个 ID 只支持一个起始点。`legacy` 使用 6 维 Kalman `[x,y,w,h,vx,vy]` 与 Hungarian，默认 `0.7*(1-IoU)+0.3*relative_area_difference`，阈值 0.7、max_age=15。它是依据说明重建的 baseline，不是找回的旧代码。默认 baseline 的检测来自 Trackpy；`detector.type: yolo` 并指定本地 TEM 权重可切换检测器。

## Pipeline / 配准和坐标

读取与 hash → 空间预处理 → 背景平移配准 → 指定 ID 定位/关联 → 质量标记 → CSV/统计 → 原图覆盖层及视频。

右侧背景 ROI 用于 phase/ECC 配准交叉检查。原图坐标 `center_x/center_y` 始终保留，原点左上、x 向右、y 向下，单位 px。`x_background_relative_px/y_background_relative_px` 单独减去累计背景位移；一旦某步失效，后续累计配准坐标置空，避免把不完整累计值当完整校正。背景相对运动不等于已证明的载物台漂移。

## Run Tracking / 可复现命令

输出路径必须是新目录或空目录：

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --input data\raw\sample.mp4 --config config\default.yaml --seeds config\selected_particles.json --output results\experiment_01
```

切换方法加 `--method lk`、`--method trackpy` 或 `--method legacy`。确认科学标定后加 `--acquisition-fps 数值 --nm-per-pixel 数值`。CLI 配置中也可设置轨迹历史长度、圆圈厚度、显示 bbox，以及配准 ROI。未标定时物理单位列为空，不能把 `speed_px_per_video_s` 当实验速度。

## Run Training / 可选 YOLO baseline

此部分没有在样例上训练：未找到旧数据集、标签、TEM 权重；主追踪无需执行这一步。

先准备独立目录 `images/train`、`images/val`、`labels/train`、`labels/val` 和 `dataset.yaml`（`names: [particle]`）。标签每行 `class_id xc yc w h`，坐标归一化。原标签格式变化时先确认中心/角点约定，不直接转换。

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py data\yolo\dataset.yaml --output results\dataset_validation.json
```

校验器只读检查图像、配对标签、类别、框越界、空标注和路径重复。空标签会警告，以便核实是否为负样本。它不能自动确认粒子语义或检测不同文件名的近重复视频帧。

YOLO 建议在单独环境安装官方匹配的 torch/torchvision 与 ultralytics；不要在主环境混装 `opencv-python` 和 `opencv-python-headless`。检索时 Ultralytics 为 8.4.164、PyTorch 为 2.14.0，仅为官方包注册表快照，不代表本机 CUDA 组合已验证。安装成功后保存该训练环境自己的 `pip freeze` 和 CUDA 信息。

```powershell
python scripts\train_detector.py --data data\yolo\dataset.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --output results\yolo_training
```

这个命令可能下载初始化权重。训练脚本先校验数据，再固定随机种子，并按 CUDA 可用性选择 GPU/CPU。训练完成后把真正的 TEM 权重路径写入配置；推理入口不会自动下载 COCO 权重充当 TEM 检测器。

## Evaluation / 测试与实测

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\run_comparison.py --output results\comparison_new
.\.venv\Scripts\python.exe scripts\evaluate_tracking.py --help
```

对照覆盖默认方案、去掉配准、去掉高斯去噪、去掉中心细化、严格反向门限、搜索范围 12/20/40、LK、Trackpy、legacy 四种代价。默认搜索范围为 28。初始化帧不计入定位误差；所有方法使用相同选点和视频。

样例 **131/156 个目标帧有观测、25 个待复核**。这衡量输出可用程度，**不是 84% 准确率**。早期 48 个近似参考点上，默认平均中心误差约 7.07 px，与 Trackpy/legacy 约 7.10 px 接近，不能宣称全面胜出。正式 HOTA/IDF1/MOTA/mAP 留空。进一步验证需专家逐帧身份标注和独立视频。

## Output Format / 输出

| 文件 | 内容 |
|---|---|
| `annotated.mp4` | 绿色 ROI、黄色相邻观测轨迹、ID；底部新增 40 px 状态栏 |
| `tracks.csv` | 每个已初始化 ID 的逐帧记录；缺测仍保留一行，坐标留空 |
| `review_required.csv` | 缺测/质量不足帧及原因，不代表其余帧已由人复核 |
| `track_statistics.csv` | 观测数、缺测数、相邻观测路径长度、首末净位移 |
| `trajectories.png`、`particle_trajectories/` | 总轨迹及每粒轨迹/坐标曲线 |
| `drift.csv` | 背景位移、phase/ECC 质量、逐步与累计有效性 |
| `config.yaml`、`seeds.json` | 本次完整参数与人工锚点 |
| `metrics.json`、`summary.md`、`logs.txt` | 覆盖率、耗时、未评估项及运行日志 |
| `input_metadata.json`、`environment.json`、`source_manifest.json` | 输入 hash、解码信息、依赖版本、源文件 hash |

CSV 主要字段解释见 [数据字典](docs/data_dictionary.md)。圆圈和 bbox 是选定跟踪窗口，`roi_area_px2` 不等于分割面积；固定 ROI 不能用于推断颗粒长大。缺测、人工校正的跳变不加入路径长度，故路径长度是被观测连续片段的累计，而非真实完整路径。

## Known Limitations / 使用边界

- 相邻外观相似颗粒、接触、强形变和暗对比消失时仍有认错风险；质量门限只能筛出部分错误。
- 第 3 号后半段丢失明显，其余颗粒也存在待核查片段。丢失不等于物理消失，更不等于自动识别了融合/分裂。
- 不做精确 mask、粒子真实面积变化、扩散系数或原子结构事件推断。压缩视频不足以单独支撑高精度物理结论。
- 目前离线全帧加载，默认灰度帧内存上限 1024 MB；页面上传上限 512 MB。长视频应分段或增加内存配置，跨片段 ID 需人工管理。
- 默认样例参数不保证适合新 TEM 条件。搜索范围、半径和配准 ROI 应先在可核验的帧上调整，再在独立视频验证。

## Reproducibility / 项目结构

```text
config/              参数和选点
data/raw/            原始视频副本
data/annotations/    明确来源的参考点
src/tem_tracker/     读取、定位、配准、关联、分析、可视化
scripts/             追踪、界面、校验、训练、评估与对照
ui/                  本地中文选点页面
tests/               合成运动、缺测、校正、配准和数据完整性检查
docs/                研究、比较、实测和数据字典
results/selected_particles/   最终样例
results/comparison/           对照数据
```

固定 Python/NumPy/OpenCV 种子 17，OpenCV 单线程，每次保存输入 hash 和参数；前后 hash 相同才标记完成。逐次结果不会覆盖。不同 CPU、包版本或视频解码实现仍可能产生微小数值差异，应以自己的环境快照记录。
