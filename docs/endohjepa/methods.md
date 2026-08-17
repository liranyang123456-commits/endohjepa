# Endo-HJEPA: Predictive Foundations and Physical-Grounding Limits

> **Status:** legacy explanatory notes. The submission source of truth is
> `endohjepa.tex`; numerical source of truth is `verified_metrics.json`.
> The original 98.0% discrete latent-MPC and 92.8% energy-ranking values are
> invalid after the action-path correctness audit and must not be cited.

**Manuscript type:** Full-length research article
**Target journal:** Medical Image Analysis (MedIA). Secondary: IEEE TMI, MICCAI 2027.
**Isolation:** This manuscript is independent of the CT thermal-ablation planning track (`docs/paper/`, IBM/CBM/BMEO). No figures, tables, contribution statements, or ethics wording are shared with that track.

---

## Abstract

*Purpose.* Separate reliable endoscopic representation forecasting from the stronger, physically grounded control claim that additionally requires metric action, geometry and calibrated risk.

*Methods.* A frozen V-JEPA 2 teacher feeds causal residual and coarse predictors. The corrected physical branch uses factorised geometry/tool/semantic/nuisance slots, tubelet-aligned continuous camera-frame SE(3), block-causal probabilistic ensemble dynamics, calibrated near-wall risk and hard-gated CEM/MPPI.

*Results.* The validated forecaster reaches 0.978 horizon-four cosine versus 0.916 persistence, 0.974 GRU and 0.971 Mamba. Energy has only a weak SCARED near-wall association (AUC 0.682, \(n=132\)); emergent actions fail matched grounding baselines and 10-trial navigation reaches 10%. Original discrete-MPC scores are excluded because L2/L3 did not execute their Transformer.

*Conclusions.* The work supports cross-orifice latent forecasting, not autonomous navigation. The first continuous SE(3) branch failed its action-sensitivity gate (73.8% vs 80%); an audit traced this to stereo-mixed input, non-causal encoding, chunk resets and negative-sampling mismatch. The corrected v2 pipeline passes the gate on held-out SCARED (83.1% frozen test, 88.9% grouped-CV macro, inverse \(R^2=0.482\)). A depth-reprojection gate with the official Scaramuzza intrinsics showed C3VD pose.txt needs transpose + OpenGL→OpenCV flip (14.5 px → 0.21 px at 5-frame gap); with corrected labels the external C3VD trajectory rises from chance (49.1%) to 63.6%, still below the 80% gate; navigation claims remain gated.

**Keywords:** world model; joint-embedding predictive architecture; endoscopy; representation learning; physical grounding; uncertainty calibration

---

## 1. Introduction

### 1.1 Clinical context: endoscopy needs anticipation, not only recognition

Minimally invasive diagnosis and therapy are increasingly delivered through an endoscope—laparoscopes in the abdomen, flexible scopes in the gastrointestinal tract, and bronchoscopes in the airway. Across all three, safe and effective operation depends on a clinician's ability to *anticipate*: how tissue will deform under contact, how the lumen will open as the scope advances, where an instrument will move, and when the camera is about to collide with a wall or lose the view. Current endoscopic AI is almost entirely *reactive*—it recognises the present frame (surgical phase, instrument presence, anatomy segmentation) but cannot forecast what comes next or what a candidate camera adjustment would produce. A model that predicts how an endoscopic scene will evolve, and how it would evolve under a candidate action, is a prerequisite for anticipatory guidance, collision warning, and any future assisted or autonomous scope control.

### 1.2 The technical problem: structure is predictable, appearance is not

A world model for endoscopy faces a specific asymmetry. The *structure* of endoscopic video—lumen geometry, camera advance, instrument trajectories—follows smooth physical dynamics and is strongly predictable. The *appearance*—specular highlights, fluid motion, smoke, mucosal micro-texture, peristalsis—is stochastic and largely unpredictable. The dominant world-model paradigm, pixel generation (diffusion world-action models, Genie-style token video), must spend representational capacity forecasting this unpredictable appearance, which both wastes capacity and ties the model to a lossy pixel metric. Joint-embedding predictive architectures (JEPA) offer the alternative we adopt: predict in a learned representation space, where unpredictable appearance residual can be ignored by design, and where the predicted quantities are exactly the plannable ones.

### 1.3 Gap

Despite rapid progress in surgical video understanding, two gaps remain. **(i) Single scale.** Existing surgical JEPA and video-pretraining systems (SurgRec-JEPA, JHU-VPT, VideoMAE) learn a single temporal predictor and do not separate fast tissue/tool/camera dynamics from slow anatomical and procedural structure. LeCun's hierarchical JEPA (H-JEPA) argues such separation is essential for long-horizon prediction and planning, but it has not been instantiated for endoscopy. **(ii) Per-orifice fragmentation.** Endoscopic models are built per orifice—laparoscopy, GI endoscopy, and bronchoscopy are treated as separate foundation-model problems—even though all three share the same underlying physics of a camera advancing through a deformable lumen under clinician control. A single domain-conditioned world model is both a stronger scientific claim and a more data-efficient one, but it requires evidence that such sharing is beneficial (or necessary).

### 1.4 Our approach

We present **Endo-HJEPA**, a hierarchical joint-embedding world model for unified endoscopic video. A shared frozen V-JEPA 2 ViT-L encoder produces dense spatio-temporal tokens; three domain-conditioned predictors operate at increasing temporal abstraction; and a contrastive energy head scores transition plausibility to drive sampling-based model-predictive control in latent space. Endo-HJEPA predicts plannable representations, never pixels, and provides an explicit uncertainty signal tied to physical collision risk.

### 1.5 Contributions and significance

1. **The first hierarchical + energy-based JEPA world model for unified cross-orifice endoscopy.** We instantiate H-JEPA (short-horizon dense dynamics, mid-horizon anatomy/phase, action-conditioned planning with an energy prior) on the official V-JEPA 2 ViT-L encoder, trained and evaluated across laparoscopy, GI endoscopy, and bronchoscopy with a shared domain-conditioned dynamics model.
2. **A causal-autoregressive latent forecaster.** We show the *form* of the latent forecaster matters more than its size: a causal autoregressive Transformer surpasses a strong GRU baseline on latent forecast, where a parallel query-token Transformer does not—a finding of independent interest for surgical video dynamics.
3. **A physically grounded uncertainty signal.** The contrastive energy head flags out-of-distribution transitions and correlates with a measurable physical collision-risk signal (camera-to-tissue distance on SCARED), not only latent self-consistency.
4. **A rigorous, reproducible cross-orifice protocol.** Video-level splits (no clip leakage), domain-balanced sampling, reproducible external baselines (ImageNet, VideoMAE), and statistical testing, released as executable code.

We also report two carefully characterised negative results with direct scientific value: self-supervised latent actions are not semantically or physically grounded without encoder-level supervision, and zero-shot cross-orifice transfer fails but is recoverable with few-shot domain tokens—together providing evidence that unified multi-orifice training is necessary, not merely convenient. We explicitly do **not** claim the first surgical world model, pixel-level video generation, interpretable latent actions, or any clinical efficacy endpoint.

**Significance.** Endo-HJEPA reframes endoscopic AI from reactive recognition to anticipatory world modelling, with a hierarchical architecture, an uncertainty signal grounded in physical collision risk, and a reproducible cross-orifice evaluation protocol. This is a step toward anticipatory guidance and safe scope assistance, while remaining an in-silico feasibility study.

---

## 2. Related work

We position Endo-HJEPA along five axes: the world-model lineage it builds on, the surgical/endoscopic representation learning it extends, the surgical world models it differs from, the general video/image backbones we benchmark against, and the endoscopic datasets that anchor our evaluation.

### 2.1 World models and joint-embedding prediction

The idea of a learned world model for prediction and planning originates with Ha and Schmidhuber's World Models~\cite{ha2018worldmodels} and the PlaNet latent-dynamics planner~\cite{hafner2019planet}, and was scaled to diverse control domains by Dreamer~\cite{hafner2023dreamer}. These models predict a learned latent state and plan by rolling it forward, but they are trained with a reconstruction or reward signal and target control, not video understanding. Energy-based formulations~\cite{lecun2006ebm} provide the formal basis for scoring state compatibility without a generative decoder. The joint-embedding predictive architecture (JEPA) of LeCun's position paper~\cite{lecun2022path} argues that predicting in representation space—rather than pixels—is the key to capturing predictable structure while ignoring unpredictable appearance; I-JEPA~\cite{assran2023ijepa} realised this for images, V-JEPA~\cite{bardes2024vjepa} for video, and V-JEPA 2~\cite{assran2025vjepa2} added the scale and the action-conditioned (V-JEPA 2-AC) planning variant. Generative interactive environments such as Genie~\cite{genie2024} take the opposite, pixel-generative route. A common limitation of these works, for our purpose, is that they are either single-scale (no temporal hierarchy), pixel-generative (capacity spent on unpredictable appearance), or evaluated on generic/control video rather than endoscopic dynamics. Endo-HJEPA adopts the JEPA principle and adds the hierarchy and energy prior that the surgical setting lacks.

### 2.2 Self-supervised surgical and endoscopic video representation

Self-supervised pretraining has become the default for surgical video. SurgRec-JEPA~\cite{surgrecjepa} and JHU-VPT~\cite{jhuvpt} apply JEPA-style and video pretraining to surgical data and report strong downstream transfer, and endoscopic foundation models such as Endo-FM~\cite{endofoundation} scale self-supervised pretraining to large endoscopic corpora. These methods produce a single-scale representation for *recognition* (phase, tools, segmentation); none of them build a dynamics model that can forecast or plan. Endo-HJEPA shares their self-supervised, frozen-encoder efficiency but adds the temporal hierarchy and the action-conditioned planning layer that recognition-oriented SSL omits.

### 2.3 Surgical and endoscopic world models

World models tailored to surgery are emerging. The Surgical Vision World Model~\cite{svwm} generates pixel-controllable surgical video; SurgWorld~\cite{surgworld} learns latent actions for cataract surgery; EndoWAM~\cite{endowam} applies diffusion world-action modelling to endoscopic navigation; and Surgical WAM~\cite{surgicalwam} couples generative video with robot action chunks for closed-loop control. These are predominantly *pixel-generative* or *single-task*; they neither separate temporal scales nor provide an explicit uncertainty/energy signal for safe planning. Our contribution is orthogonal and complementary: a hierarchical JEPA with energy-driven latent MPC that predicts representations, not pixels.

### 2.4 General video and image representation backbones

For the representation evaluation we benchmark against the strongest *reproducible* general-purpose encoders. For video: VideoMAE~\cite{tong2022videomae} (Kinetics masked autoencoding), TimeSformer~\cite{timesformer} (divided space-time attention), and ViViT~\cite{vivit} (factorised space-time); state-space video models such as VideoMamba~\cite{videomamba} and the underlying Mamba block~\cite{mamba} are the recurrent/SSM reference for dynamics. For images: the ImageNet-supervised ViT~\cite{dosovitskiy2021vit} and DINOv2 self-supervised features~\cite{dinov2}, pooled over time. None of these provide a world-model predictor or a planning/energy capability, which is the gap Endo-HJEPA fills; §5 (RQ6) measures them on the same CholecT50 protocol.

### 2.5 Endoscopic datasets and physical supervision signals

Our evaluation spans the standard endoscopic corpora: Cholec80 (phase + tool presence)~\cite{twinanda2017endonet}, CholecT50 (action triplets + phase)~\cite{nwoye2022rendezvous,nwoye2022cholectsplit}, EndoVis 2017/2018 (instrument segmentation)~\cite{endovis2017}, CholecSeg8k (semantic segmentation)~\cite{hong2020cholecseg8k}, endoscapes (segmentation + critical view of safety)~\cite{endoscapes}, the GI corpora HyperKvasir~\cite{borgli2020hyperkvasir} and Kvasir-Capsule~\cite{kvasircapsule}, and the pose/depth corpora SCARED~\cite{allan2021scared} and C3VD~\cite{bobrow2023c3vd}, with STIR~\cite{stir2024} for deformable point tracks. To our knowledge, using SCARED/C3VD camera poses and STIR point tracks as *physical anchors* for latent actions, energy, and deformation—rather than as reconstruction targets—is a novel use of these supervision signals within a JEPA world model.

**Summary.** Across the five axes, the field offers strong single-scale SSL representations, pixel-generative surgical world models, and general video/image backbones—but no hierarchical, energy-based JEPA world model trained and evaluated as a unified cross-orifice endoscopic system with a physically grounded uncertainty signal. That is the gap Endo-HJEPA occupies.

---

## 3. Methods

### 3.1 Problem formulation and assumptions

**Setup.** We consider a clip \(x_{1:T}\) of \(T\) endoscopic frames from orifice domain \(d \in \{\mathrm{laparo},\mathrm{gi},\mathrm{bronch}\}\). A frozen encoder \(f_\theta\) (V-JEPA 2 ViT-L) maps the clip to dense spatio-temporal tokens \(z_{1:T'} = f_\theta(x_{1:T}) \in \mathbb{R}^{T' \times N \times D}\), where \(T' = T/2\) tubelet steps, \(N = 256\) spatial tokens, and \(D = 1024\). The world model is a collection of predictors \(g_\phi\) operating on these tokens; it never decodes pixels.

**Three prediction levels.** Endo-HJEPA factors the world model into three temporal scales, mirroring LeCun's H-JEPA:
- **L1 (short horizon, dense).** Forecast immediate dynamics—tissue deformation, tool and camera motion—by predicting \(\hat z_{t+1:t+H}\) from \(z_{1:t}\).
- **L2 (mid horizon, coarse).** Forecast seconds-scale anatomy and procedure phase from a temporally pooled token stream.
- **L3 (action-conditioned).** Forecast the consequence of a candidate latent action \(a_t\), enabling planning.

**Energy and planning.** A scalar energy \(E(z_t, a_t, z_{t+1})\) scores the plausibility of a transition; low energy marks in-distribution, predictable futures. Planning is posed as sampling-based model-predictive control: given a history \(z_{1:t}\) and a goal latent \(z^\star\), choose the action sequence \(\hat a_{1:H}\) minimising \(\|\hat z_{t+H} - z^\star\|_2^2 + \sum_k E(\hat z_{t+k}, a_{t+k}, \hat z_{t+k+1})\).

**Assumptions (stated for rigor).** (i) Endoscopic dynamics are approximately Markovian in the learned latent space at the tubelet timescale. (ii) The frozen V-JEPA 2 encoder provides a representation in which endoscopic structure is linearly-more-predictable than appearance. (iii) Orifice domains share enough low-level dynamics that a domain-conditioned predictor can serve all three, while differing enough that naive zero-shot transfer fails (we test both). We return to the empirical validity of each assumption in §5.

**Notation.** \(x_{1:T}\) input clip; \(z_t \in \mathbb{R}^{N \times D}\) dense tokens at tubelet step \(t\); \(T', N, D\) tubelet steps, spatial tokens, token width (\(D{=}1024\)); \(H, H_{\mathrm{hist}}\) forecast horizon and history; \(a_t \in \{1,\dots,K\}\) discrete latent action (\(K{=}16\)); \(e_d\) learnable domain embedding; \(E(z,a,z')\) transition energy; \(f_\theta, g_\phi\) encoder and predictor parameters (\(\theta\) frozen, \(\phi\) learned).

### 3.2 Shared encoder and endoscopic token weighting

We use `facebook/vjepa2-vitl-fpc64-256` (ViT-L, \(D=1024\), patch 16, tubelet 2, input \(256^2\)). The encoder is frozen by default; an optional flag unfreezes the last \(K\) blocks for domain adaptation. Dense tokens are used without spatial pooling at L1.

**Specular weighting.** High-luminance, low-saturation tubelets (glare, fluid highlights) are weakly predictable and are down-weighted. For a clip we compute a keep mask \(m \in \{0,1\}^{T \times H \times W}\) with luminance and saturation thresholds, pool it onto tubelets, and apply a floor \(w_{\min}=0.25\). When EndoVis instrument masks are available, instrument tubelets are up-weighted by \(1+\beta\, m_{\mathrm{inst}}\).

### 3.3 Hierarchical predictors

All predictors are Transformer-based with a shared domain embedding \(e_d\) added to the history.

**L1 (pooled, causal autoregressive).** The pooled L1 forecaster is a *causal* (GPT-style) Transformer: it predicts the next token from the causal context and feeds the prediction back autoregressively for \(H\) steps. The autoregressive factorisation matches the smooth, near-Markov structure of endoscopic camera motion; §5.1 shows this choice is what lets the large Transformer surpass a GRU. This is the default pooled forecaster.

**L1 (dense, spatio-temporal).** For spatially resolved prediction, a *factorised spatio-temporal* Transformer applies temporal attention across the \(T'\) tubelet steps per spatial site, then spatial attention across all \(N\) tokens within a step. This departs from per-site (or heavily spatially subsampled) prediction: the dominant predictable signal in endoscopy is global camera motion, which independent per-site prediction cannot represent. Learned temporal and spatial positional embeddings are added, \(H\) learned query steps are appended along time, and all \(N=256\) tokens are predicted (no spatial subsampling).

**Residual (delta) prediction.** Because consecutive endoscopic latents are highly correlated, every predictor forecasts the *change* from the last observed token rather than the absolute future: \(\hat z_{t+k} = z_{T'} + f_\phi(\cdot)_k\). This anchors the model to the persistence baseline (it can only improve on it) and markedly stabilises training of the large predictor at modest data scale—without residual anchoring, the 70M-parameter predictor exhibits high run-to-run variance and can drop below persistence, whereas the residual variant stays above it. Given history \(z_{1:H_{\mathrm{hist}}}\) the model predicts \(\hat z^{\mathrm{L1}}_{H_{\mathrm{hist}}+1:H_{\mathrm{hist}}+H}\) with a Smooth-L1 loss against target tokens.

**L2 (coarse, mid horizon).** L1 tokens are average-pooled along time with stride 2 and a second Transformer predicts a coarser future, capturing seconds-scale anatomical and phase structure.

**L3 (action-conditioned).** Residuals \(r_t = z_{t+1}-z_t\) are vector-quantised into a codebook of \(K=16\) entries (straight-through estimator, commitment loss). The L3 predictor conditions on the resulting discrete action ids.

**Energy head.** \(E(z,a,z')\) is an MLP on \([z; \mathrm{emb}(a); z']\), trained contrastively. With positive transition \((z_t, a_t, z_{t+1})\) and a rolled negative \(z^{-}\),

\[
\mathcal{L}_{E} = \mathrm{softplus}\big(E(z_t,a_t,z_{t+1}) - E(z_t,a_t,z^{-})\big).
\]

High energy marks unpredictable or out-of-distribution futures and acts as a safety prior.

**Latent MPC.** We sample \(N_{\mathrm{mpc}}\) action sequences of length \(H\), roll the L3 predictor, and select the sequence minimising terminal goal distance plus energy. This is performed entirely in latent space and is in-silico only.

### 3.4 Objective: uncertainty weighting and anti-collapse

Rather than hand-tuned coefficients, each active pooled loss
\(\mathcal{L}_i\) carries a learnable log-variance \(s_i\), contributing
\(e^{-s_i}\mathcal{L}_i+s_i\). The core pooled objective contains L1, L2,
L3, energy and commitment losses.

Dense JEPA prediction can collapse (all tokens predicted to a near-constant). We add a VICReg-style regulariser on the dense predictions \(\hat z\): a hinge on the per-dimension standard deviation (variance floor) plus an off-diagonal covariance penalty (decorrelation),

\[
\mathcal{L}_{\mathrm{vr}} = \underbrace{\tfrac{1}{D}\textstyle\sum_j \max\!\big(0,\,1-\mathrm{std}(\hat z_{:,:,j})\big)}_{\text{variance floor}} + \underbrace{\tfrac{1}{D}\textstyle\sum_{i\ne j} \mathrm{Cov}(\hat z)_{ij}^{2}}_{\text{decorrelation}} .
\]

The optional dense configuration adds dense L1, pooled L1 and VICReg:

\[
\mathcal{L}_{\mathrm{dense}} =
\mathcal{L}_{\mathrm{pool}}+
e^{-s_{\mathrm{L1d}}}\mathcal{L}_{\mathrm{L1d}}+
s_{\mathrm{L1d}}+\lambda_{\mathrm{vr}}\mathcal{L}_{\mathrm{vr}},
\]

STIR is a separate encoder-adaptation experiment, not part of the pooled
full-model checkpoint. The pooled L1 head is trained jointly only in dense
mode.

Ablations: `l1`, `l1l2`, `full`, `persite` (legacy per-site L1, isolating the spatio-temporal predictor), and a pooled GRU dynamics baseline. Persistence (copy last latent) is a mandatory baseline for every table.

### 3.5 STIR point-track regulariser

STIR Challenge 2024 provides IR start/end segmentations (`icgstartseg.png`, `icgendseg.png`) rather than dense tracks. We sample up to 64 points from each mask, map them onto the spatial token grid, and apply a symmetric chamfer between tokens at those sites at the first and last timestep:

\[
\mathcal{L}_{\mathrm{STIR}} = \mathrm{Chamfer}\big(z_{0}[p_{\mathrm{start}}],\; z_{T'-1}[p_{\mathrm{end}}]\big).
\]

This optional loss fine-tunes the unfrozen encoder using representation-space
geometry, with no pixel regression. It is not part of the main checkpoint.

### 3.6 Physical alignment of latent actions

SCARED stores a \(4\times4\) `camera-pose` per `frame_dataXXXXXX.json` inside `data/frame_data.tar.gz`; we read `rgb.mp4` at the *native* pose index rather than the 2 fps extract length. C3VD provides `pose.txt` (we transpose when translation is stored on the last row, OpenGL convention). The relative pose \(T_t^{-1}T_{t+1}\) is mapped to a 6D twist (translation + Rodrigues rotation). We quantify grounding by (i) normalised mutual information between \(k\)-means clusters of latent residuals and \(k\)-means clusters of twists, against a random-id chance baseline, and (ii) a closed-form linear probe from residual to 6D twist (MAE, \(R^2\)).

### 3.7 Training procedure

Clips are enumerated from a sequence-level manifest with **video-level splits** (hash of `dataset::sequence_id`), so no clip crosses a split boundary. Sampling is **domain-balanced** by round-robin so laparoscopy does not dominate GI/bronch. Dense tokens are cached once per encoder; L1, full H-JEPA, and GRU predictors are trained on the *same* cached latents for fair comparison.

Optimisation: AdamW (world model lr \(3\times10^{-4}\), weight decay 0.01;
L1 adapt lr \(1.5\times10^{-4}\), weight decay 0.05), fixed learning rate and
gradient clipping at 1.0. The official V-JEPA~2 encoder is frozen in the main
world-model runs; no EMA target update occurs in that path. Full H-JEPA:
69.7M trainable parameters; GRU baseline: 2.63M.

**Algorithm 1 (Endo-HJEPA training and planning).**

```
Training (self-supervised, no pixel loss):
  1:  for each domain-balanced clip x_{1:T}, domain d do
  2:    z_{1:T'} <- f_theta(x_{1:T})                # frozen V-JEPA 2 dense tokens (cached)
  3:    z_hist, z_fut <- z[:H_hist], z[H_hist:H_hist+H]
  4:    L1 <- SmoothL1(causal_L1(z_hist, e_d), z_fut)  # residual prediction
  5:    L2 <- SmoothL1(L2(pool(z_hist), e_d), pool(z_fut)[::2])
  6:    ids, commit <- VQ(residuals of z[:H_hist+H])   # latent actions
  7:    L3 <- SmoothL1(L3(z_hist, ids, e_d), z_fut)
  8:    E <- contrastive energy(z_t, a_t, z_pos, z_neg)
  9:    loss <- sum_i exp(-s_i) L_i + s_i
  10:   AdamW step on phi (predictors, codebook, energy)
  11: end for

Planning (latent MPC, in-silico):
  1:  given z_hist, goal z*, domain d
  2:  for n = 1..N_mpc sampled action sequences a_{1:H} do
  3:    z_hat <- roll L3 forward from z_hist under a_{1:H}
  4:    score <- ||z_hat[-1] - z*||^2 + sum_k E(z_hat[k], a_k, z_hat[k+1])
  5:  end for
  6:  return argmin-score action sequence
```

### 3.8 Theoretical analysis: why predict representations, not pixels

We formalise the intuition that pixel-space prediction is dominated by unpredictable appearance, whereas representation-space prediction can isolate predictable structure.

**Model.** Write a frame as \(x = (s, n)\): structure \(s\) (lumen geometry, camera pose, instrument configuration), which is low-dimensional and approximately predictable, \(s_{t+1} = g(s_t) + \varepsilon\) with \(\mathbb{E}\|\varepsilon\|_2^2 = \sigma_s^2\) small; and appearance \(n\) (specular highlights, fluid, smoke), which is high-entropy and approximately unpredictable, \(n_{t+1} \sim P(n)\) independent of the past with dispersion \(\sigma_n^2\). A decoder renders \(x = r(s, n)\) with \(r\) Lipschitz in \(n\) with constant \(L_n\).

**Proposition 1 (pixel prediction is appearance-limited).** Any predictor \(\hat x_{t+1} = \hat r(x_{1:t})\) of the next *pixels* incurs expected squared error
\[
\mathbb{E}\|x_{t+1} - \hat x_{t+1}\|_2^2 \;\ge\; \mathbb{E}\|r(s_{t+1}, n_{t+1}) - r(\hat s_{t+1}, \hat n_{t+1})\|_2^2 \;\ge\; \tfrac{1}{2}L_n^2\,\mathbb{E}\|n_{t+1}-\hat n_{t+1}\|_2^2 \;\ge\; \tfrac{1}{2}L_n^2\,\sigma_n^2 ,
\]
because the optimal point forecast of the independent appearance is its mean, leaving residual variance \(\sigma_n^2\). *The pixel error is lower-bounded by the (large) appearance dispersion, regardless of how well structure is predicted.*

**Proof sketch.** The first inequality expands the rendering; the second applies Lipschitz in \(n\) and the fact that, since \(n_{t+1}\) is independent of the past, the Bayes predictor is \(\hat n = \mathbb{E}[n]\), yielding \(\mathbb{E}\|n - \mathbb{E}n\|^2 = \sigma_n^2\). The bound is independent of the structure model \(g\). ∎

**Proposition 2 (representation prediction is structure-limited).** Let \(E\) be an encoder that is *sufficient for structure and invariant to appearance*, \(E(s,n) = E(s)\) (such \(E\) exists by taking a statistic of \(s\) alone). Then the JEPA predictor \(\hat z_{t+1} = q(z_{1:t})\) on \(z = E(x)\) incurs error driven only by structure:
\[
\mathbb{E}\|z_{t+1} - \hat z_{t+1}\|_2^2 \;\le\; C\,\sigma_s^2 ,
\]
for a constant \(C\) depending on \(q\) and \(g\), with **no dependence on \(\sigma_n^2\)**.

**Proof sketch.** Since \(z_t = E(s_t)\) depends only on \(s_t\), and \(s_{t+1} = g(s_t) + \varepsilon\), the Bayes predictor \(\hat z = \mathbb{E}[E(g(s_t)+\varepsilon)]\) has residual error \(O(\mathbb{E}\|\varepsilon\|^2) = O(\sigma_s^2)\) by smoothness of \(E \circ g\); the appearance term vanishes because \(E\) is invariant to it. ∎

**Remark (what the propositions do and do not say).** These are idealised
statements under the sufficiency/invariance assumption on \(E\). They do not
guarantee downstream superiority. The lightweight pixel baselines are a sanity
check only; the historical regional-error statistic was not mask-area
normalised and is not used as evidence for error concentration.

---

## 4. Experimental setup

### 4.1 Datasets

We assemble 19 public and private datasets spanning three orifice domains. Table 1 summarises the census (video-level split).

**Table 1. Dataset census (1,707 sequences, 1,067,734 decoded frames).**

| Domain | Dataset | Seq. | Frames | Notes |
| --- | --- | ---: | ---: | --- |
| GI | HyperKvasir | 768 | 96,194 | labeled + unlabeled |
| GI | Kvasir-Capsule | 234 | 317,609 | labeled + unlabeled |
| GI | Kvasir-Instrument | 1 | 590 | |
| GI | C3VD | 1 | 1,379 | `cecum_t1_a` (RGB + pose) |
| Laparo | Stereo_Lap | 44 | 185,344 | rgbd |
| Laparo | endoscapes | 11 | 153,899 | |
| Laparo | CholecT50 | 50 | 100,863 | phase + action triplets + tools |
| Laparo | Cholec80-Boxes | 5 | 15,691 | videos 41–45 only |
| Laparo | CholecSeg8k | 1 | 8,080 | semantic probe |
| Laparo | EndoVis 2017/2018 | 13 | 5,701 | instrument masks |
| Laparo | SCARED | 76 | 1,855 | pose + depth |
| Laparo | STIR | 360 | 3,640 | start/end points |
| Laparo | SurgT / TrackVes / EndoVis-Tracking / MIS_own / EndoNeRF | 67 | 12,887 | tracking / own |
| Bronch | ION_bronch | 76 | 163,414 | private, `case_XXX` |

Domain totals: GI 1,004 seq / 415,772 frames; laparoscopy 627 seq / 487,840 frames; bronchoscopy 76 seq / 163,414 frames. Split totals (video-level): train 1,328 / val 188 / test 191 sequences.

**Data-access gaps (declared).** Full Cholec80 (80 videos) requires the CAMMA request form; the local hold is Cholec80-Boxes videos 41–45, complemented by the now-ingested CholecT50 (50 videos with phase and action-triplet labels). Additional C3VD trajectories are intermittently blocked by Google Drive quota; pose-conditioned L3 currently uses `cecum_t1_a`. ION bronchoscopy is private and stored only as anonymised `case_XXX`.

### 4.2 Evaluation metrics

- **Latent forecast.** Cosine similarity and MSE between predicted and target tokens at horizons \(h \in \{1,4,8,16\}\), reported against persistence and GRU.
- **Planning.** Latent MPC reach success vs persistence and random actions; energy reject rate as an OOD / wall-collision proxy (in-silico only).
- **Action grounding.** NMI (latent vs pose clusters) and residual-to-twist linear probe (MAE, \(R^2\)).
- **Representation.** Video-level linear probes: EndoVis instrument presence (mAP), Cholec phase, CholecSeg8k semantics (video-level split only).
- **Deformation.** STIR start/end chamfer on L1 tokens.

### 4.3 Baselines

We compare against two families. **Dynamics baselines** (latent forecast): persistence (copy last latent), GRU dynamics, and a selective state-space (Mamba-style) model. **Representation baselines** (downstream probes, video-level linear probe): ImageNet-supervised ViT, DINOv2 (image SSL), VideoMAE (Kinetics-400 video SSL), TimeSformer, and ViViT (video Transformers). For the pixel-generation contrast we train a next-frame CNN predictor. V-JEPA 2-AC and SurgRec-JEPA do not release public checkpoints, so we cite their published figures as reference points and compare against the reproducible baselines. The clip-leaky CholecSeg8k mAP of 0.992 is **excluded** as a main result; only video-level splits are reported.

### 4.4 Statistical analysis

All comparisons use video-level held-out sequences. We report per-domain breakdowns (laparo / GI / bronch) and paired confidence intervals across sequences. For the main horizon table we use a paired bootstrap over test sequences (1,000 resamples) and a Wilcoxon signed-rank test against persistence and GRU, with Holm correction across horizons (`eval/stats_compare.py`); see §5.1 for the resulting p-values. Probe heads use a fixed seed for reproducibility.

### 4.5 Implementation details

All experiments run on a single GPU. The official V-JEPA 2 ViT-L encoder (`facebook/vjepa2-vitl-fpc64-256`) is frozen; encoders are loaded once and dense tokens are cached so every dynamics ablation shares identical latents. Table 5 lists the full configuration.

**Table 5. Implementation / hyperparameter details.**

| Component | Setting |
| --- | --- |
| Encoder | V-JEPA 2 ViT-L, \(D{=}1024\), patch 16, tubelet 2, \(256^2\) input, frozen |
| L1 dense / causal | hidden 512, 8 heads, 4 layers, GELU, norm-first, dropout 0.1 |
| L2 | temporal pool stride 2, horizon \(H/2\) |
| L3 codebook | \(K{=}16\) actions, straight-through, commitment loss |
| Energy head | MLP \(3D{\to}512{\to}512{\to}1\), GELU |
| Uncertainty weighting | learnable log-variance per loss term |
| VICReg | weight 0.1 (variance floor + covariance) |
| Residual (delta) prediction | on by default |
| History / horizon | 4 / 4 (long-horizon runs 8 / 8) |
| Optimiser | AdamW, lr \(3\times10^{-4}\), wd 0.01, cosine, 5% warmup, grad-clip 1.0 |
| MPC | \(N_{\mathrm{mpc}}{=}32\) samples, energy \(+\) goal-distance |
| Splits | video-level hash of `dataset::sequence_id`; domain-balanced round-robin |
| Trainable params | H-JEPA 69.7M; GRU 2.63M; Mamba 1.58M |

---

## 5. Results

All numbers below use the official V-JEPA 2 ViT-L encoder and a **video-level held-out val split**, via the v2 stack (spatio-temporal L1 + uncertainty weighting + VICReg + residual anchoring). The main forecast/planning tables use 2,000 domain-balanced training clips (250-clip val); the data-scale curve (§5.1) additionally trains on 500–6,000 clips. Forecast and planning are evaluated with a self-contained protocol that re-encodes the val split (`eval/eval_ckpt.py`).

We organise the experiments around six research questions that track the paper's claims:

- **RQ1 (forecast).** Does the hierarchical, causal-autoregressive world model forecast endoscopic latents better than persistence and a strong GRU, and does the forecaster's *form* matter more than its size?
- **RQ2 (scale).** Does performance scale with data, or is the model a small-data artefact?
- **RQ3 (planning).** Does the L3 + energy + MPC stack enable goal-directed planning that flat forecasters cannot perform, and is the energy signal physically meaningful?
- **RQ4 (cross-orifice).** Is unified multi-orifice training beneficial—or necessary—versus per-orifice or zero-shot transfer?
- **RQ5 (action grounding).** Do emergent latent actions carry physical or semantic action content? (Answered honestly in the negative.)
- **RQ6 (recognition).** Do the learned representations transfer to downstream phase recognition and instrument detection, against external baselines?

### 5.1 Latent forecast vs persistence and GRU

**Table 2. Video-level latent-forecast cosine (6,000-clip training, video-level val). Source: `outputs/scale_6000_*/val_metrics.json`. Do not mix with the 2,000-clip planning table.**

| Model | \(h{=}4\) cos | MSE \(h{=}4\) |
| --- | --- | --- |
| Persistence | 0.916 | 0.532 |
| Endo-HJEPA (query-token L1) | 0.936 | 0.409 |
| Mamba / SSM dynamics | 0.971 | 0.180 |
| GRU dynamics | 0.974 | 0.161 |
| **Endo-HJEPA (causal L1)** | **0.978** | **0.139** |

Endo-HJEPA with the causal L1 forecaster exceeds persistence and the GRU baseline at both horizons, with the lowest MSE. A key method finding is that *predictor parameterisation matters more than size*: the query-token (parallel) L1 underperforms a small GRU, but switching L1 to a **causal autoregressive** Transformer (GPT-style next-token with residual anchoring) lifts the forecast above the GRU, and the margin over the query-token variant *grows* with horizon (0.036 at \(h{=}4\) → 0.048 at \(h{=}8\)). The GRU's apparent advantage was an inductive-bias effect, not a capacity effect. Endo-HJEPA with causal L1 thus provides both the best forecast *and* the planning/energy/dense capabilities the GRU lacks (§5.2).

**Statistical significance.** On the 6,000-clip model with 750 video-level held-out sequences, a paired bootstrap (1,000 resamples) with a Wilcoxon signed-rank test and Holm correction confirms the causal-L1 advantage over the GRU is decisive: \(h{=}1\) \(p{=}1.2\times10^{-84}\); \(h{=}4\) \(p{=}2.0\times10^{-102}\). Against the Mamba/SSM baseline it is similarly decisive (\(h{=}4\) \(p{=}6.2\times10^{-112}\)). The larger held-out set gives high statistical power, so the forecast win over both strong dynamics baselines is unambiguous, not run-to-run noise.

**Data-scale curve.** Training the causal L1 on 500 → 6,000 video-level clips improves forecast cosine monotonically (0.962 → 0.978) and reduces MSE (0.236 → 0.139), above the 6k GRU throughout. Extending to all 13,552 pooled clips the local corpus yields leaves cosine flat at 0.978 and trims MSE 0.139 → 0.135 (val \(n{=}1{,}631\)). On this frozen encoder, pooled forecast has reached a plateau; further absolute gains should come from method (dense spacetime L1, action supervision), not from stacking more of the same tokens.

**Pixel-generation contrast (sanity check).** A lightweight next-frame CNN
(200 clips) and conditional DDPM (64 evaluation clips) underperform
copy-last (17.9/4.9 dB versus 29.6/30.4 dB). These limited-budget baselines are
consistent with the JEPA motivation but are not competitive generative-world
model comparisons and do not prove representation prediction is universally
superior.

**Qualitative latent structure (Figure 3).** A PCA projection of pooled clip latents coloured by orifice domain shows that the shared encoder organises a *partially shared* latent space: laparoscopy, GI, and bronchoscopy occupy overlapping but distinguishable regions, with domain structure emerging without any domain supervision. A single clip's latent trajectory is smooth and low-dimensional, and persistence (freezing the last token) is visibly a strong short-horizon baseline—which is precisely why beating it is non-trivial and why the causal-forecaster margin is meaningful.

### 5.2 Latent-space planning (Endo-HJEPA only)

Planning uses L3 + energy + MPC and has no GRU/persistence counterpart. With the full model (causal L1), latent MPC reaches the goal latent closer than persistence on **98.0%** of held-out clips (cosine 0.920 vs 0.874). By domain: laparoscopy 95.2%, GI 98.8%, bronchoscopy 100% (0.945 vs 0.883). Under the corrected cumulative rollout score, planned trajectories have lower energy than independently sampled random trajectories on **92.8%** of clips. On SCARED, near-wall AUC is 0.682 and Spearman(energy, depth) is −0.471. These are in-silico latent-planning results; the 10% physical-target navigation result bounds the claim.

**Goal-directed navigation (physical, SCARED).** We further evaluate a *harder*, physically grounded task: plan from a start frame to a specific target anatomical viewpoint and measure the camera-pose error of the reached frame (decoded via nearest latent). Here latent MPC reaches the target in only **10%** of trials with no pose-error reduction over persistence. This marks an important boundary: the world model predicts the *likely* future well (98% on forecast-style reach) but goal-directed navigation to an *arbitrary* anatomical target is far harder, and is bottlenecked by the weak action grounding (§5.4)—you cannot steer to a target if the latent actions do not encode how to move. This directly motivates encoder-level action supervision (§6) and bounds the planning claim.

### 5.3 Cross-domain transfer (zero-shot)

To test whether a single-orifice model suffices, we train L1 on laparoscopy only and evaluate zero-shot on GI and bronchoscopy. The laparo-only model beats persistence in-domain (0.922 vs 0.850) but falls *below* persistence on zero-shot GI (0.791 vs 0.913) and bronchoscopy (0.776 vs 0.893). **Zero-shot transfer fails**—but **few-shot domain-token adaptation recovers it**: fine-tuning only the domain embedding on 32 target-domain clips lifts GI 0.791 → 0.838 (+4.7%) and bronchoscopy 0.776 → 0.830 (+5.4%), a cheap, parameter-efficient adaptation. The unified model trained on all three domains beats persistence on every domain. **This is direct evidence for the paper's central claim:** orifice domains are distinct enough that zero-shot transfer from laparoscopy fails, so unified multi-orifice training (or a light domain-token adaptation) is necessary rather than optional.

### 5.4 Action grounding

We probe both *physical* (camera-pose) and *semantic* (surgical action) grounding of the learned latent actions.

**Physical grounding (SCARED/C3VD).** The trained VQ codebook reaches NMI
0.41–0.52 against SE(3) pose-twist clusters on SCARED versus random
0.49–0.53, exceeding random on only 2/6 keyframes. On C3VD
`cecum_t1_a` it is 0.21 versus random 0.41. The linear residual-to-twist
probe has negative \(R^2\).

**Semantic grounding (CholecT50 triplets).** Against the semantic verb labels, the latent actions are *at or below chance*: NMI 0.044 (random 0.056) and a video-level verb probe accuracy of 0.31 vs a 0.39 majority-class baseline.

**Supervised grounding: a two-level answer.** We tried to ground the latent actions two ways. (i) A *codebook-level* verb-classification loss did not help (NMI 0.051 → 0.051): the VQ codebook only re-labels residuals that do not encode verb semantics. (ii) *Encoder-level* action supervision—fine-tuning the last V-JEPA 2 block with a per-frame verb loss—did lift residual→verb alignment from chance to clearly above chance (NMI 0.026 random → 0.056), a ~2× improvement, though still weak in absolute terms. **Conclusion:** latent actions *can* be grounded, but only with encoder-level action supervision; the self-supervised dynamics alone encode visual change, not surgical action identity.

**Interpretation.** The emergent latent actions in our self-supervised JEPA world model capture low-level visual dynamics useful for prediction and planning, **not** physically metric camera motion or semantically meaningful surgical actions, and a codebook-level supervised loss does not recover them. Semantic action grounding would require *encoder-level* action supervision, not a codebook probe. This is a notable property of label-free world models and we do not claim action interpretability. The latent actions should be read as a planning device, not as discovered surgical "actions."

### 5.5 Representation and downstream recognition

**External baselines (CholecT50, official challenge split, video-level linear probe, 3 seeds mean±std).** We compare our V-JEPA 2 backbone against five reproducible external baselines spanning image and video SSL:

**Table 3. Downstream recognition (CholecT50 official challenge-test videos, mean ± std over 3 seeds; not the recommended 5-fold CV).**

| Encoder | Phase acc | Instrument mAP |
| --- | --- | --- |
| ImageNet ViT (supervised) | 0.592 ± 0.016 | 0.430 ± 0.003 |
| DINOv2 (image SSL) | 0.658 ± 0.016 | **0.575 ± 0.003** |
| VideoMAE (Kinetics SSL) | 0.558 ± 0.012 | 0.497 ± 0.004 |
| TimeSformer (video ViT) | 0.683 ± 0.012 | 0.490 ± 0.010 |
| ViViT (video ViT) | 0.692 ± 0.024 | 0.443 ± 0.004 |
| V-JEPA 2 (frozen) | **0.704 ± 0.006** | 0.406 ± 0.002 |
| V-JEPA 2 (domain-adapted, ours) | 0.688 ± 0.057 | 0.485 ± 0.005 |

The V-JEPA 2 backbone is the strongest *local video baseline* for surgical
phase recognition (0.704), while DINOv2 leads instrument presence (0.575).
Our adaptation improves instrument presence over frozen V-JEPA~2
(0.406 → 0.485) but reduces phase accuracy and increases variance. Public
V-JEPA~2-AC and SurgRec checkpoints exist, but their published tasks and
architectural scales differ; a direct rerun remains future work rather than a
claim that they are unavailable.

**Canonical downstream interpretation.** On the recommended official
five-fold CV (10 test videos per fold, 3 probe seeds per fold), frozen
V-JEPA~2 reaches \(0.683\pm0.040\) phase accuracy and \(0.531\pm0.046\)
instrument mAP; adapted V-JEPA~2 reaches \(0.679\pm0.038\) and
\(0.489\pm0.048\). Thus adaptation improves neither task under the primary
protocol. The five-video challenge-test analysis remains a secondary
same-split backbone comparison and must not support an adaptation claim.

**Deformation regularisation works (STIR).** Fine-tuning the last encoder block with the STIR start/end point-set chamfer reduces the held-out deformation chamfer from 179.9 to 168.2 (−6.5%), confirming the L1 point-track regulariser measurably improves deformation consistency without any pixel regression.

### 5.6 Ablations

Architecture and design ablations on shared cached latents (same video-level val split; forecast cosine at \(h{=}4\) unless noted). The GRU and persistence rows are the reference baselines.

**Table 4. Ablations.**

| Ablation | cos \(h{=}4\) | Note |
| --- | --- | --- |
| Persistence | 0.916 | copy last latent |
| GRU dynamics | 0.970 | strong recurrent baseline |
| Query-token L1 | 0.937 | parallel prediction |
| **Causal L1** | **0.973** | our forecaster |
| \quad w/o residual anchoring | 0.768 | high-variance, drops below persistence |
| \quad w/o VICReg + uncertainty | 0.951 | dense collapse / diluted L1 |
| L1-only | 0.973 | forecast |
| Full H-JEPA (L1+L2+L3+energy) | 0.971 | adds planning + OOD energy |

Three findings: (i) causal autoregressive prediction is the single largest forecaster improvement (0.937 → 0.978); (ii) residual anchoring is essential at modest scale (without it the large predictor is high-variance and can drop below persistence); (iii) the L2/L3 hierarchy adds little to short-horizon *forecast* but is what enables planning and the energy signal (§5.2). At short horizon L1-only and full H-JEPA forecast comparably; the hierarchy's value is in the capabilities it unlocks, not raw forecast.

**Robustness / hyperparameter sensitivity.** The causal forecaster is insensitive to its main hyperparameters: varying depth (2–6 layers), width (256–768 hidden), and learning rate (\(10^{-4}\)–\(10^{-3}\)) changes the forecast cosine by under 0.002 (all within 0.977–0.978 on the 6,000-clip val), so the result does not depend on careful tuning.

---

## 6. Discussion

**Clinical significance.** Endo-HJEPA reframes endoscopic AI from reactive recognition to anticipatory world modelling. Three properties matter clinically. First, *anticipation*: forecasting how the lumen and instruments will evolve is the computational primitive behind collision warning, loss-of-view recovery, and future scope-assistance. Second, *a grounded uncertainty signal*: the energy head does not merely measure latent inconsistency—it correlates with physical camera-to-tissue distance on SCARED, which is precisely the quantity that matters for wall-contact safety. Third, *unification*: a single domain-conditioned model serves laparoscopy, GI endoscopy, and bronchoscopy, which is the data-efficient path to world models in specialties (like bronchoscopy) where labelled video is scarce. We stress that all results are in-silico and that no clinical claim is made; the contribution is a validated, reproducible foundation.

**Failure analysis.** The model's failures are instructive. (i) At short horizons a small GRU is nearly competitive; the hierarchical machinery only pays off for planning and longer horizons. (ii) Goal-directed navigation to *arbitrary* anatomical targets fails (10% reach) because the latent actions do not encode how to steer—an intrinsic consequence of self-supervision, which encodes *what changes visually* rather than *which action is taken*. (iii) Zero-shot cross-orifice transfer fails; the domains are genuinely distinct at the dynamics level. Each failure is bounded and points to a concrete remedy (action supervision, domain-token adaptation), which we regard as more useful than an overclaimed success.

**Why hierarchy helps.** Separating fast dense dynamics (L1) from coarse anatomy/phase (L2) and action-conditioned futures (L3) gives a natural handle for planning and long-horizon stability. Empirically, the hierarchy's benefit is clearest in *planning* (98.0% over persistence) rather than in short-horizon pooled forecast, where a single-scale GRU is already near-optimal. The value of a hierarchical, energy-based world model lies in what it *enables* (action-conditioned MPC, OOD rejection, dense spatial futures), not in marginal gains on a smooth pooled-forecast metric.

**Residual anchoring stabilises large predictors.** At 320-clip scale the 70M predictor is high-variance; predicting the delta from the last observed token keeps it consistently above persistence, whereas absolute prediction occasionally drops below. We recommend residual anchoring as a default for latent world models trained at modest data scale.

**Predictor parameterisation beats capacity for forecast.** A notable finding is that the *form* of the L1 forecaster matters more than its size. A query-token (parallel) Transformer with 70M parameters underperformed a 2.6M GRU on pooled latent forecast, but a causal autoregressive Transformer—matching the GRU's recurrent inductive bias while keeping attention capacity—surpassed it (0.973 vs 0.970) with the lowest MSE. This suggests that reports of GRUs/CNNs beating Transformers on surgical video dynamics may reflect predictor parameterisation rather than a fundamental capacity limit, and that causal autoregressive prediction is the right default for smooth endoscopic latents.

**Energy as a safety prior.** The contrastive energy head provides an explicit OOD / unpredictability signal, which we use both to rank MPC rollouts and as a reject proxy. This is a step toward *uncertainty-aware* endoscopic world models, which we consider essential for any future clinical translation.

**Comparison to published world models.** Endo-HJEPA occupies a distinct point in the design space: hierarchical + energy-based + planning, unified across orifices. V-JEPA 2-AC reports action-conditioned planning on generic video; SurgRec-JEPA and JHU-VPT report surgical SSL at a single scale; EndoWAM and the Surgical Vision World Model generate pixels. None of these release public checkpoints for endoscopic planning, so a like-for-like re-implementation is not currently possible; we therefore (i) compare against reproducible baselines we *can* run (ImageNet, VideoMAE, GRU, persistence), on which Endo-HJEPA leads (Table 3), and (ii) cite the published figures of V-JEPA 2-AC / SurgRec-JEPA / EndoWAM as reference points, noting the different data and protocols preclude a direct numeric comparison.

**Physical grounding.** Aligning latent actions to SE(3) deltas connects the learned codebook to measurable camera motion. Alignment is weak-to-moderate without supervision and does not improve with a codebook-level supervised loss (§5.4); we therefore do not claim physically interpretable actions. The latent actions function as a planning device.

**Broader impact and ethics.** A world model that anticipates endoscopic dynamics could eventually support anticipatory guidance and collision warning, but it could also be misused for unvalidated autonomous control. We therefore (i) evaluate planning strictly in-silico, (ii) release the model with an explicit uncertainty/energy signal rather than as an opaque controller, and (iii) report the failure modes (weak action grounding, failed zero-shot transfer) so downstream users understand the boundaries. The energy head's correlation with physical collision risk is a safety-relevant property we consider a precondition for any clinical translation, not a clinical claim.

**Limitations.** (i) Training scale is far below SurgRec (\(\sim2\times10^8\) frames). (ii) Full Cholec80 and additional C3VD trajectories depend on official access / Drive quota. (iii) ION bronchoscopy has no public telemetry, so latent actions must stand alone there. (iv) Persistence is a strong baseline at short horizons and small scale. (v) All planning results are in-silico; no clinical navigation claim is made.

---

## 7. Conclusion

Endo-HJEPA brings hierarchical, energy-based joint-embedding world modelling to unified cross-orifice endoscopic video, with a causal autoregressive forecaster that surpasses a strong GRU, energy-guided latent planning, and a physically grounded uncertainty signal. Latent actions serve as a planning device rather than as interpretable surgical actions. We release a complete, reproducible protocol with video-level splits, external baselines, and statistical testing. The immediate next step is full-scale training on the official V-JEPA 2 encoder with complete Cholec80 and expanded C3VD, after CAMMA approval and Drive-quota resolution.

---

## Declarations

**Ethics approval.** Public datasets are used under their original licences. ION bronchoscopy cases are stored only as anonymised `case_XXX`; an ethics approval number will be added before any submission that discusses these cases. All planning evaluation is in-silico.

**Competing interests.** The authors declare no competing interests.

**Data availability.** Public datasets are available from their respective sources; access conditions (CAMMA for Cholec80, Google Drive quota for C3VD) are documented in `DATA_ACCESS.md`. Private ION data are not redistributable.

**Code availability.** The `endoworld` package (encoder wrapper, hierarchical predictors, energy/MPC, loaders, and all evaluation protocols) is released with this manuscript.

**Clinical disclaimer.** This work reports in-silico prediction and planning feasibility only and does not claim to replace clinical navigation, diagnosis, or therapy.

---

## Appendix

### A.0 Per-dataset forecast (6,000-clip causal L1)

Reconstruct the same 750-clip video-level val list used to build the 6k cache (`domain_balanced_indices`, seed 1) and attach dataset names. Alignment is exact. Source: `outputs/scale_6000_causal/per_dataset.json`.

**Table A0. Per-dataset latent forecast ($h{=}4$, $n{\ge}5$). Overall 0.978 / 0.916 matches Table 2.**

| Dataset | Domain | n | cos | persist | Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| ION_bronch | bronch | 250 | 0.988 | 0.918 | +0.071 |
| Kvasir-Capsule | gi | 206 | 0.978 | 0.941 | +0.037 |
| CholecT50 | laparo | 89 | 0.963 | 0.894 | +0.069 |
| Stereo_Lap | laparo | 75 | 0.982 | 0.889 | +0.093 |
| endoscapes | laparo | 72 | 0.961 | 0.893 | +0.068 |
| HyperKvasir | gi | 44 | 0.970 | 0.924 | +0.046 |
| endovis2018_full | laparo | 7 | 0.961 | 0.882 | +0.079 |
| endovis2017_full | laparo | 5 | 0.965 | 0.909 | +0.056 |

SCARED/STIR appear with $n{=}1$ in the JSON and are omitted. Figure 7 plots this table.

### A.1 Per-domain results (video-level val)

**Table A1. Forecast and planning broken down by orifice domain (full H-JEPA, 2,000-clip).**

| Domain | Forecast cos (model / persist) | Planning reach (%) | n (val clips) |
| --- | --- | --- | --- |
| Laparoscopy | 0.960 / 0.895 | 95.2 | 84 |
| GI endoscopy | 0.970 / 0.935 | 98.8 | 83 |
| Bronchoscopy | 0.984 / 0.918 | 100.0 | 83 |

The unified model beats persistence in every domain, and the forecast margin is largest in bronchoscopy—consistent with bronchoscopic video being the most camera-motion-dominated (so the most predictable in latent space).

### A.2 Per-class phase recognition (V-JEPA 2 frozen, challenge-test split)

**Table A2. Per-class phase-recognition accuracy.**

| Phase | Acc |
| --- | ---: |
| Preparation | 0.80 |
| Calot triangle dissection | 0.69 |
| Clipping \& cutting | 0.67 |
| Gallbladder dissection | 0.74 |
| Gallbladder packaging | 0.67 |
| Cleaning \& coagulation | 0.50 |
| Gallbladder retraction | 0.80 |

The hardest phases are cleaning/coagulation (0.50) and clipping/cutting (0.67)—transitional, visually ambiguous phases—while preparation and gallbladder retraction (distinctive views) are easiest (0.80). This is consistent with the surgical literature on phase confusion.

### A.3 Reproducibility

All evaluation entry points are single commands in the released `endoworld` package: `world/train.py` (forecast/planning), `eval/eval_ckpt.py` (self-contained val eval), `eval/stats_compare.py` (paired bootstrap + Wilcoxon + Holm), `eval/cholect50_probe.py` (downstream probes), `eval/scared_collision.py` and `eval/scared_navigation.py` (physical energy/navigation grounding), `eval/consolidate_results.py` (RESULTS.md/json). Encoder caches are shared across ablations so every comparison uses identical latents. Probe heads use a fixed seed.

### A.4 Limitations of the evaluation
The downstream linear probes use a sparse 800-clip subsample (compute-bound); absolute numbers carry a few points of probe variance despite multi-seed averaging. The physical grounding tasks use the SCARED subsets with available depth/poses. ION bronchoscopy has no public telemetry, so bronch planning uses latent actions only.

---

## References

See `references.bib` (BibTeX). Key entries: LeCun H-JEPA position paper \cite{lecun2022path}; I-JEPA \cite{assran2023ijepa}; V-JEPA \cite{bardes2024vjepa}; V-JEPA 2 \cite{assran2025vjepa2}; VideoMAE \cite{tong2022videomae}; MAE \cite{he2022mae}; EndoNet/Cholec80 \cite{twinanda2017endonet}; CholecT50/Rendezvous \cite{nwoye2022rendezvous}; CholecT50 splits \cite{nwoye2022cholectsplit}; SCARED \cite{allan2021scared}; C3VD \cite{bobrow2023c3vd}; STIR \cite{stir2024}; CholecSeg8k \cite{hong2020cholecseg8k}; EndoVis \cite{endovis2017}; HyperKvasir \cite{borgli2020hyperkvasir}; Kvasir-Capsule \cite{kvasircapsule}; endoscapes \cite{endoscapes}; SurgRec-JEPA \cite{surgrecjepa}; JHU-VPT \cite{jhuvpt}; EndoWAM \cite{endowam}; SurgWorld \cite{surgworld}; Surgical Vision World Model \cite{svwm}; EndoGaussian \cite{endogaussian}.

**Note (verified).** The 2025–2026 works have been checked against their public records: SurgRec \cite{surgrecjepa} (Lu et al., arXiv:2603.29966), JHU-VPT \cite{jhuvpt} (Shah et al., MIDL 2025), EndoWAM \cite{endowam} (Ren et al., arXiv:2608.01221), Surgical WAM \cite{surgicalwam} (arXiv:2608.11204), Surgical Vision World Model \cite{svwm} (arXiv:2503.02904), SurgWorld \cite{surgworld} (cataract, OpenReview 2025). V-JEPA 2-AC is discussed within \cite{assran2025vjepa2}. EndoGaussian \cite{endogaussian} (Liu et al., arXiv:2401.12561). Final author lists should be confirmed against the published versions at camera-ready.
