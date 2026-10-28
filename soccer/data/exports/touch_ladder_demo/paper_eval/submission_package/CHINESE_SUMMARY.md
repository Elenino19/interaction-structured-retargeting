# 投稿包装：冻结 Soccer Full 的可视化呈现（未重跑 IK）

冻结哈希已核验（预览 `c904d2ed…039e`，robot_a/b、ball 与 `frozen_inputs.json` 一致）。源视频、打包片段、机器人预览均为 **24 fps / 144 帧，无帧率转换**：流水线第 \(i\) 帧 = 原片 \(203 + i/24\) 秒。

## 图怎么用

- **论文若不能使用源片**（当前无再分发许可）：用 `fig_robot_events.pdf` + `fig_interaction_timeline.pdf`。
- **作者本地/若取得许可**：用 `local_only/fig_source_to_robot.pdf`（上源下机器人，六列事件）。带水印版在 `local_only/fig_source_to_robot_watermarked.pdf`。
- **结果表**：`fig_results_reuse.pdf`（复用已有数字，非新实验）。

六列事件：f0 A 右脚开球 → f15 B 右脚接球 → f21 B 左脚踢球（肢体角色切换）→ f26 球在通道中（B→A 传递）→ f42 A 右脚接球（主体角色切换）→ f45 A 右脚回做（同脚接踢）。

## 交互状态如何落到 Soccer（论文可直接改写）

标注 \(\mathcal{E}\) 给出每拍的主体、接/踢球脚与时刻，以及 6 条跨体 kick→receive 传递。据此生成通道时刻表；每个 G1 **独立** IK（事件窗只改该机器人的权重/支撑/踢球速度项）。球轨迹是 IK **之后** 按双方脚点与通道合成的规定路径，不是接触物理，也不是联合优化。Event IK 在固定球上不改变 13/13、6/6；它改善踢球脚速度 RMSE（0.66→0.20 m/s）并增大 jerk。链一致性主要来自标注 + 站位 + 球的后处理合成。

## 可写进论文的数字（单序列）

- Full：13/13 主动肢体关联，6/6 标注传递链，中心门控 τ=0.05 m 为 6/6。
- 固定球 Event IK ON/OFF：关联与链不变。
- 独立对象感知基线：同样 13/13、6/6，表面误差优于 Full；求解器之间无通信。

## 不要写

- 跨体联合优化 / 求解器自动发现事件标注 / 物理触球 / 真机 / 多样本统计。

## 视频

- 冻结 Full：`data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4`（与 `paper_eval/videos/full.mp4` 同哈希）
- 源|机器人并排（仅本地）：`local_only/source_robot_side_by_side.mp4`
- 源片本身不可随论文打包。
