# 项目进度与复现说明

更新日期：2026-10-01  
当前阶段：本地 Touch Ladder 双人足球动作重定向概念验证已完成并通过主观验收。

## 1. 当前结论

项目已经跑通“单目双人视频 → 人体伪标签 → 双 G1 动作 → 球/标志桶场景 →
可视化视频”的完整链路。当前验收版本为：

- `data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4`
- 同内容标准预览：
  `data/exports/touch_ladder_demo/g1_ball_cones_preview.mp4`
- 机器可读结果清单：
  `data/exports/touch_ladder_demo/result_manifest.json`

该版本保持原始动作主体，仅对世界坐标中的站位和横移时机做了局部修正，使机器人
在传球后可见的小跳/换位阶段完成横向移动，减轻持续平移造成的“滑冰感”。用户已
确认当前效果满意。

## 2. 已完成里程碑

### 2.1 素材与合规

- 完成公开候选素材筛选和许可审计；
- 实现公开来源的 fail-closed 验证；
- 确认当前 7MLC / MZC CONTROLLO 素材只作本地演示，不作数据集发布；
- 确认公开“界外球—停球—回传”样本仍缺失，SoccerTrack-v2 仍需
  `HF_TOKEN`。

审计记录：`data/manifests/source_audit.json`。

### 2.2 视频预处理

- 采用原视频 203.0–209.0 秒窗口；
- 960×544、24 fps、144 帧；
- YOLOv8x 检测左右两名参与者；
- 以 1.8 倍边界框扩张并保证最小 420×500 区域；
- 输出两路 420×544 裁剪，并遮挡另一名参与者，降低 GVHMR 身份切换风险。

平均人物检测高度为 A 约 292.44 px、B 约 333.25 px。

### 2.3 双人单目动作恢复

- 对 A、B 两路裁剪分别运行官方 GVHMR 静态相机模式；
- 两路结果均为 144 帧；
- 保留官方 `hmr4d_results.pt`、预处理缓存、incamera 和 global 预览；
- 通过 SMPL-X 前向运动学导出关节，再分别调整朝向、落地并设置 2.8 m
  名义间距。

这里的 2.8 m 是展示和重定向用几何尺度，不表示从视频中恢复出的真实世界距离。

### 2.4 球、标志桶和事件恢复

- YOLO 检测足球，144 帧中 129 帧得到检测，其余帧插值；
- HSV 分割恢复 5 个标志桶；
- 将 GVHMR 脚踝/脚尖投影回源图像；
- 用交替参与者约束的动态规划得到接触事件初值；
- 将接球帧/脚与传球帧/脚分开表达，支持“右脚停球、左脚出球”等动作；
- 结合机器人脚位和指定缝隙生成最终球轨迹。

当前事件：

1. A：第 0 帧右脚出球（第 0 帧左脚只作为片段开头上下文）；
2. B：第 15 帧右脚停球，第 21 帧左脚出球；
3. A：第 42 帧右脚停球，第 45 帧右脚出球；
4. B：第 71 帧右脚接/传；
5. A：第 90 帧左脚接/传；
6. B：第 110 帧右脚接/传；
7. A：第 132 帧左脚接/传。

六次传球依次使用缝隙索引 `[3, 4, 5, 4, 3, 2]`，对应机器人 Y
坐标 `[-0.25, -0.75, -1.25, -0.75, -0.25, +0.25] m`。第一和
第三次传球使用脚到脚单直线减速轨迹；其余传球先进入对应缝隙直线段，并仅在出脚
或接球端完成必要的横向换线。

### 2.5 双 G1 重定向

- 使用 Unitree G1 29-DoF URDF；
- A、B 独立优化，最后映射到共享展示场景；
- 当前运行使用 340 次迭代；
- 以人体关键点匹配为主，同时加入根部正则、速度/加速度、支撑脚防滑、脚底离地、
  骨盆起伏、上身速度、躯干方向、肩线和传球脚速度约束；
- 按事件切换发力脚与支撑脚权重；
- 最终仅对根部世界 Y 坐标施加分段平滑站位偏移，不改写已经恢复的关节动作。

主要超参数：

- 学习率 0.04；
- 关键点 1.0，根部正则 0.35；
- 基础速度 0.035，基础加速度 0.08；
- yaw 0.4，hip yaw 0.06；
- 关节速度 0.02，加速度 0.04；
- 支撑脚滑移 6，脚底地面 6；
- 骨盆高度 18，骨盆垂直速度 40；
- 上身速度 8，胸部朝向 6，肩线 4；
- 传球脚速度 24。

当前轨迹中保存的缩放系数为 A `0.9470989763`、B `0.9246479386`。
最终目标函数值分别为 `0.051696`、`0.034268`；这只是同一优化器内的收敛诊断，
不能作为跨方法精度指标。

### 2.6 渲染与版本留档

- 采用 MuJoCo EGL 离屏渲染；
- 足球半径 0.11 m；
- 球位置按规定轨迹逐帧写入 mocap body，旋转由相邻位移和球半径积分；
- 标志桶不参与碰撞，避免规定轨迹被引擎碰撞改写；
- 输出 960×720、24 fps、144 帧、6 秒视频；
- 每次重写预览或机器人轨迹前自动复制到 `archive/`。

文档同步前的旧 `metadata.json`、`prep.json`、`cooperative_motion.npz` 和
`incam_pair_preview.mp4` 也已保留在
`data/exports/touch_ladder_demo/archive/documentation_sync/`。

## 3. 当前内部诊断

以下数字均从当前验收文件重新计算，不是动捕真值或正式 benchmark：

- 六次传球相对最近标志桶的最小中心距离依次约为
  `0.221, 0.250, 0.185, 0.260, 0.264, 0.257 m`；
- 第一脚在标志桶线附近的 Y 坐标约 `-0.283 m`，第三脚约
  `-1.190 m`，其余四脚分别为 `-0.750, -0.750, -0.250,
  +0.250 m`；
- 第一、第三脚的移动段方向跨度均接近 0°，即单一直线；
- B 第 15 帧右脚停球时球心—脚尖距离约 0.108 m；
- B 第 21 帧左脚出球时约 0.095 m；
- A 第 42 帧右脚停球和第 45 帧右脚出球时分别约 0.082 m 和
  0.101 m；
- 骨盆横向速度绝对值的 95 分位数：A 约 2.46 m/s，B 约
  2.26 m/s；最大值分别约 4.08 m/s、3.59 m/s。

球半径为 0.11 m，球心—脚尖距离略小于球半径表示点模型进入球体表面，不等同于
机器人网格发生相同深度的穿模。完整数值见 `result_manifest.json`。

## 4. 产物位置

### 输入和预处理

- 原视频：`$SOCCER_SOURCE_VIDEO`
- 源片段：`data/clips/touch_ladder_demo_source.mp4`
- A/B 裁剪：`data/clips/touch_ladder_demo_player_a.mp4`、
  `data/clips/touch_ladder_demo_player_b.mp4`
- 预处理元数据：`data/clips/touch_ladder_demo_prep.json`

### GVHMR

- A：
  `data/gvhmr/touch_ladder_demo/player_a/touch_ladder_demo_player_a/`
- B：
  `data/gvhmr/touch_ladder_demo/player_b/touch_ladder_demo_player_b/`
- 每个目录中的核心文件为 `hmr4d_results.pt`，并包含 `1_incam.mp4`、
  `2_global.mp4`、`bbx.pt`、`vit_features.pt`、`vitpose.pt`。

### 中间表示

- 双人参数打包：`data/exports/touch_ladder_demo/cooperative_motion.npz`
- SMPL-X 关节：`data/exports/touch_ladder_demo/smplx_joints.npz`
- 源图像脚部轨迹：`data/exports/touch_ladder_demo/source_foot_tracks.npz`
- 球/标志桶数组：`data/exports/touch_ladder_demo/ball_cones.npz`
- 接触事件：`data/exports/touch_ladder_demo/ball_cones.json`

### G1 和视频

- A/B 机器人：
  `data/exports/touch_ladder_demo/g1/robot_a.npz`、
  `data/exports/touch_ladder_demo/g1/robot_b.npz`
- 比例：`data/exports/touch_ladder_demo/g1/scales.json`
- 场景：`data/exports/touch_ladder_demo/g1/soccer_scene.xml`
- 当前视频：
  `data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4`
- 上一版本：
  `data/exports/touch_ladder_demo/comparison/current_straight_first_clean_second.mp4`
- 恢复基线：
  `data/exports/touch_ladder_demo/comparison/restored_straight_gap_pass_baseline.mp4`
- 视频历史：`data/exports/touch_ladder_demo/archive/`
- 机器人历史：`data/exports/touch_ladder_demo/g1/archive/`

## 5. 复现步骤

项目依赖两个外部环境：

- GVHMR：`$GVHMR_ROOT`，Python
  `$GVHMR_ROOT/.venv/bin/python`
- G1/MuJoCo：`$ISR_JOIN_ROOT`，Python
  `$ISR_JOIN_ROOT/.venv/bin/python`

在仓库根目录依次运行：

```bash
python3 scripts/prepare_touch_ladder_demo.py

$GVHMR_ROOT/.venv/bin/python scripts/run_gvhmr_pair.py \
  touch_ladder_demo \
  --player-a data/clips/touch_ladder_demo_player_a.mp4 \
  --player-b data/clips/touch_ladder_demo_player_b.mp4 \
  --gvhmr-root $GVHMR_ROOT \
  --python $GVHMR_ROOT/.venv/bin/python

$GVHMR_ROOT/.venv/bin/python scripts/pack_touch_ladder_demo.py
$GVHMR_ROOT/.venv/bin/python scripts/export_smplx_joints.py \
  --gvhmr-root $GVHMR_ROOT
$GVHMR_ROOT/.venv/bin/python scripts/export_source_foot_tracks.py \
  --gvhmr-root $GVHMR_ROOT
$GVHMR_ROOT/.venv/bin/python scripts/recover_ball_cones.py

$ISR_JOIN_ROOT/.venv/bin/python \
  scripts/retarget_dual_g1_soccer.py --iterations 340

MUJOCO_GL=egl $ISR_JOIN_ROOT/.venv/bin/python \
  scripts/render_g1_soccer.py
```

注意：

1. 若输入片段改变，先删除 `data/gvhmr/touch_ladder_demo/`，否则官方 GVHMR
   可能复用旧缓存；输入不变时无需重复运行 GVHMR。
2. 当前 `export_smplx_joints.py` 和 G1 URDF/XML 默认值含本机绝对路径，迁移机器时
   必须通过参数或代码修改路径。
3. 重跑 `recover_ball_cones.py` 会恢复检测/动态规划的初始结果；当前验收球轨迹还
   依赖后续 `retarget_dual_g1_soccer.py` 的脚位吸附和几何修正。
4. 渲染器使用规定球轨迹；MuJoCo 在这里负责模型加载和可视化，不负责推导传球
   接触动力学。
5. 不要挂载或写入存在 NTFS 完整性问题的 `all` 卷。

## 6. 验证状态

截至 2026-10-04 已完成：

- 4 个 `unittest` 均通过；
- 所有 Python 脚本和测试通过 `py_compile`；
- 当前视频及关键 NPZ/JSON 的帧数、尺寸、字段和路径交叉核对；
- 文档中旧的 194.5–200.0 秒、132 帧和“尚未运行 GVHMR”表述已移除；
- `incam_pair_preview.mp4` 已重新打包为当前 144 帧版本。
- 已完成 Basic IK、+ support、+ event semantics、Full 四组 340 次迭代
  消融、统一指标和 2×2 同步视频，结果见 `docs/ABLATION_EXPERIMENTS.md`。

## 7. 下一阶段

论文下一步建议：

1. 增加更多有许可序列，报告跨动作和跨视角结果；
2. 组织多名观察者进行自然度、接触可信度和球轨迹合理性评分；
3. 针对 Full 中较高的骨盆垂直加速度代理和最坏帧穿地继续优化；
4. 若研究目标要求物理可执行性，再加入动力学、接触冲量、力矩和稳定性验证；
5. 将硬编码的事件和缝隙修正逐步替换为可学习或可优化的统一时空模型。
