# 磁盘清理候选（需你确认后才删除）

E 盘剩余约 **99 GB**。工作区 `E:\World_Agent_Enoscopy` 内最大占用：

| 路径 | 约大小 | 说明 | 建议 |
| --- | ---: | --- | --- |
| `datasets/` | **561 GB** | 原始内镜数据，论文依赖 | **不要删** |
| `outputs/` | **35 GB** | 训练缓存 / 权重 / JSON | 下面分项确认 |

## 建议删除（可再生成，不丢主表数字）

主表数字都在 `outputs/*/val_metrics.json` 和 `docs/endohjepa/verified_metrics.json`，删缓存不影响已写入论文的结果。

| 路径 | 约大小 | 原因 |
| --- | ---: | --- |
| `outputs/cache_2000/latents_cache.pt` | **18.0 GB** | 稠密 2000-clip 缓存，最大单项；可用 `--cache-only` 重编 |
| `outputs/e2e_smoke/` | **1.4 GB** | e2e 冒烟，与 `vjepa2_adapted/vjepa2_adapted.pt` 重复 |
| `outputs/endohjepa_vjepa2/latents_cache.pt` | 384 MB | 早期小规模缓存 |
| `outputs/endohjepa_vjepa2_l1/latents_cache.pt` | 384 MB | 同上 |
| `outputs/endohjepa_vjepa2_gru/latents_cache.pt` | 384 MB | 同上 |
| `outputs/vjepa2_l1/latents_cache.pt` | 384 MB | 同上 |
| `outputs/endohjepa_smoke_*` | 小 | smoke，勿引用 |

以上合计约 **21 GB**。

## 建议保留（主实验）

- `outputs/scale_6000_causal/` — 主 forecast / Wilcoxon
- `outputs/scale_6000_gru/`、`scale_6000_mamba/`、`scale_6000_query/` — 对照
- `outputs/p2000_full_causal/` — planning / SCARED / grounding
- `outputs/vjepa2_adapted/` — 下游 probe 与适应编码器（1.4 GB 权重要留）
- `outputs/t16_transfer_laparo/` — 零样本 / few-shot
- 各 `val_metrics.json`、`eval_ckpt.json`、`cholect50_probe_*.json`

## 可选（另一条论文线，确认后再动）

| 路径 | 约大小 | 说明 |
| --- | ---: | --- |
| `outputs/ct_cache_ion/` | 653 MB | CT 消融线，与 Endo-HJEPA 无关 |
| `outputs/ablation_*` / `intelligent_planning*` | 视内容 | CT / 规划另一稿 |

## 如何确认

回复例如：

- `删 18GB cache_2000`
- `删建议删除那一批`
- `ct_cache 也删`
- `先不删`

确认前我不会删除任何文件。
