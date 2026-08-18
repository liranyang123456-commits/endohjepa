# Cover Letter — Medical Image Analysis

Dear Editors and Reviewers,

We submit our manuscript entitled **"Endo-HJEPA: A Hierarchical Latent World Model for Cross-Orifice Endoscopic Video Prediction with Audited Physical Grounding"** for consideration in *Medical Image Analysis*.

**The problem.** Endoscopic AI is almost entirely reactive: it recognises the present frame but cannot anticipate how the scene will evolve. Meanwhile, world-model research assumes, without testing, that predictive video features are physically grounded enough to score or plan camera motion. We show both gaps matter in practice, and we address them together.

**What we contribute.** Endo-HJEPA is a hierarchical joint-embedding world model built on a frozen V-JEPA 2 ViT-L encoder, shared across laparoscopy, gastrointestinal endoscopy, and bronchoscopy. It predicts future representations rather than pixels, with a causal residual forecaster for short horizons and a coarse predictor for mid-horizon anatomy. A block-causal branch conditions on measured continuous SE(3) camera increments, with probabilistic uncertainty, calibrated risk, and a hard safety gate.

**Why this is rigorous.** The paper is built around correctness auditing. We identified and fixed a fatal implementation defect (an encoder that was constructed but never executed), verified camera-pose conventions by depth reprojection against official intrinsics, matched counterfactual negative distributions between training and evaluation, and developed all physical-branch variants under grouped cross-validation with a frozen test set. Every number is recorded in a released metric ledger and regenerable from the released code.

**Key results.** On 19 datasets and 1,707 sequences under video-level splits, the causal forecaster reaches 0.978 cosine at horizon four versus 0.916 for persistence, 0.974 for a GRU, and 0.971 for Mamba (Holm-corrected Wilcoxon p < 1e-80). The corrected continuous SE(3) branch passes its prespecified action-sensitivity gate on held-out SCARED cases (83.1%; grouped-CV 88.9%; three seeds 85.9-86.5%). Behaviour-constrained continuous model-predictive control passes all offline navigation gates (200 trials, 51.5% reach, 29.0% pose-error reduction versus persistence). We report external C3VD generalisation (above chance, below gate) and near-wall risk calibration (fails to generalise) as characterised limits rather than claims.

**Significance.** The three validated capabilities, forecasting scene evolution, evaluating the visual consequence of a measured camera motion, and model-based navigation planning, map directly onto loss-of-view warning, camera-handling training, and navigation-assistance prototypes. The paper defines precisely where physically grounded control does and does not follow from passive video features.

**Reproducibility.** Code, the verified metric ledger, all cited result files, and figure generators are released at https://github.com/liranyang123456-commits/endohjepa. Private ION bronchoscopy data cannot be shared; all ION-derived numbers are marked.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests.

Sincerely,

Ranyang Li (corresponding author), on behalf of all authors
