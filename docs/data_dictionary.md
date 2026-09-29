# CSV 数据字典

坐标原点为图像左上，x 向右，y 向下；frame 从 0 开始。除手动输入外，数值是算法观测估计，不自动等于物理真值。

| 字段 | 含义 |
|---|---|
| frame / track_id | 视频帧号 / 用户所选粒子身份 |
| detection_id | 该输出测量的 frame:ID 标识；不等于检测器的全图候选编号 |
| center_x / center_y | 原始图像坐标，px；缺测为空 |
| radius_px / width / height | 选定圆形 ROI 半径与其窗口宽高 |
| x1 / y1 / x2 / y2 | ROI 的外接窗口，边缘目标可能超出画面 |
| roi_area_px2 | 固定 ROI 圆面积；不是粒子分割面积 |
| class_id | 单类 particle=0 |
| confidence | 主方法 NCC 相似性分数；不是已校准概率；不同方法不直接可比 |
| observed | true 表示该帧提供中心测量或人工锚点；false 表示缺测 |
| status | seed 首次人工点；manual_anchor 中途人工点；tracked 接受观测；reacquired 短缺测恢复；uncertain 未过质量检查；lost 等待人工重新确认 |
| quality_reason | 拒绝或诊断原因。缺测行可以保留被拒绝候选的质量诊断，但不保留该候选作为测量位置 |
| video_time_s | 容器播放时间，来自解码器或 frame/fps 回退 |
| acquisition_time_s | 实际采集时间 frame/acquisition_fps，仅确认采集帧率后填写 |
| timestamp / time_basis | 所采用时间及其基准；默认 video_playback_only |
| nm_per_pixel / x_nm / y_nm | 用户给定的物理标定及原图坐标换算；未标定为空 |
| dx / dy / drift_x / drift_y | 背景逐步位移与累计可用步进，px；仅 cumulative_valid=true 时累计量可信 |
| valid / cumulative_valid | 当前配准步有效 / 从首帧累计的所有配准步有效 |
| x_background_relative_px / y_background_relative_px | 减去完整可信背景位移的坐标；关闭配准或累计失效后为空 |
| step_px | 同 ID 相邻、均有观测、且当前不是人工重设点的位移模长；缺测及校正跳变不计算 |
| speed_px_per_video_s | step_px / 相邻播放时间差；不是已标定实验速度 |
| speed_nm_per_acquisition_s | step_px × nm_per_pixel × acquisition_fps，两个标定都提供后才计算 |

`track_statistics.csv` 的 coverage 是输出观测率，不是准确率；path length 只累加有效连续片段；net displacement 比较首末观测，期间可能有缺测。身份尚未核实前不把这些统计当作发表结论。
