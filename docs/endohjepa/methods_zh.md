# Endo-HJEPA 方法正文（中文提纲）

正式投稿正文为英文：`methods.md`（MedIA / MICCAI）。本稿与 `docs/paper/` 消融规划稿隔离。

**主编码器：** 官方 V-JEPA 2 ViT-L。`outputs/endohjepa/`、`outputs/vjepa_l1/` 的 scratch 9M 数字只作架构调试，**不得作主表**。

## 主张

预测可规划的表征，而不是下一帧像素。跨腔（腹腔镜 / 软镜 GI / 支气管镜）共享动力学，域 token 条件化。不声称“第一个外科世界模型”。

## 结构

1. **编码器** 冻结官方 ViT-L；可选解冻最后 K 块。高光 tubelet 降权；EndoVis mask 可上权器械 token。
2. **L1** 密空间 token 短时预测 + 时间平滑 + STIR 起止点 Chamfer（不用像素回归）。
3. **L2** 时间 stride-2 粗粒度（解剖 / 阶段）。
4. **L3** \(z_{t+1}-z_t\) 向量量化；SCARED `frame_data` / C3VD `pose.txt` 的 SE(3) 增量做 NMI 与线性探针。
5. **能量 + MPC** 对比能量；潜空间采样最低能量路径。仅 in-silico。

## 协议

视频级划分（`dataset::sequence_id` 哈希），域均衡采样。主表只引用 `outputs/endohjepa_vjepa2/`。

## 数据缺口（须在文中写明）

- 完整 Cholec80：CAMMA 申请后 `python -m endoworld.data.cholec80 --src <dir>`。本地仅 Boxes 41–45。
- C3VD 其余轨迹：Drive 限流。本地 `cecum_t1_a` 有位姿与深度，无 RGB；位姿对齐以 SCARED `rgb.mp4` 为准。
- ION：仅 `case_XXX`；投稿前需伦理批号。
