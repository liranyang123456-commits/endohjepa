# Cover Letter — Medical Image Analysis

Dear Editor-in-Chief,

We are pleased to submit our manuscript entitled

  "Endo-HJEPA: A Hierarchical Joint-Embedding World Model for Unified
   Endoscopic Video"

for consideration as a Full-Length Research Article in Medical Image Analysis.

Authors:
  Ranyang Li (Henan University of Technology; corresponding author)
  Nan Wei (Henan Provincial People's Hospital)
  Zhipeng Lin (Beihang University)
  Wufeng Liu (Henan University of Technology)
  Chao Fan (Henan University of Technology)
  Junjun Pan (Beihang University; corresponding author)

Endoscopic AI is almost entirely reactive—systems recognise the present frame
but cannot anticipate how the scene will evolve or how it would change under a
candidate camera action. Safe guidance, collision avoidance, and future scope
assistance require a world model that forecasts endoscopic dynamics. Because
endoscopic appearance is dominated by unpredictable factors (specular
highlights, fluid, smoke, peristalsis), pixel-generative world models spend
capacity on what cannot be predicted. We instead predict plannable
representations, not pixels, via a hierarchical joint-embedding predictive
architecture (H-JEPA) with a contrastive energy head for uncertainty.

Main contributions:
1. The first hierarchical + energy-based JEPA world model trained and evaluated
   as a unified cross-orifice system (laparoscopy, GI endoscopy, bronchoscopy),
   on the official V-JEPA 2 ViT-L encoder.
2. A causal-autoregressive latent forecaster that surpasses a strong GRU and a
   Mamba/SSM baseline—showing forecaster form matters more than capacity.
3. A physically grounded uncertainty signal: the energy head flags
   out-of-distribution transitions and correlates with physical camera-to-tissue
   collision risk on SCARED, not only latent self-consistency.
4. A rigorous, reproducible cross-orifice protocol (video-level splits,
   domain-balanced sampling, external baselines, statistical testing), released
   as executable code.

Key numbers (all locally verified, video-level held-out):
- Latent forecast cosine 0.978 vs 0.916 (persistence), 0.974 (GRU), 0.971
  (Mamba) at horizon 4; Wilcoxon p<1e-100 against each on 750 held-out sequences.
- Energy-guided latent planning reaches goal latents better than persistence on
  98.0% of held-out clips (100% on bronchoscopy).
- Energy head separates planned from random actions on 90.8% of clips and flags
  near-wall transitions on SCARED at AUC 0.68 (Spearman -0.47 with depth).
- V-JEPA 2 is the strongest video backbone for CholecT50 phase recognition
  (0.704, above ViViT/TimeSformer/DINOv2/ImageNet/VideoMAE); domain adaptation
  lifts instrument presence from 0.406 to 0.485 mAP.
- Forecast cosine rises monotonically 0.962→0.978 as training data scales
  500→6,000 clips.
- Honest negative results: emergent latent actions are not semantically or
  physically grounded without encoder-level supervision; zero-shot cross-orifice
  transfer fails (recoverable +4.7%/+5.4% with 32-shot domain tokens); a trained
  pixel-generation baseline underperforms copy-last-frame (4.9 vs 30.4 dB PSNR),
  empirically validating the representation-prediction premise.

We believe the work fits Medical Image Analysis's scope on machine-learning
methods for medical image computing with rigorous, reproducible evaluation. The
manuscript is original, has not been published previously, and is not under
consideration elsewhere. It is fully disjoint from our separate CT
thermal-ablation planning work. Public datasets are used under their original
licences; private bronchoscopy cases are anonymised and all planning evaluation
is in-silico. Code and the evaluation protocol will be released upon publication.

Thank you for your consideration.

Sincerely,

Ranyang Li
School of Artificial Intelligence and Big Data
Henan University of Technology
Email: lry@haut.edu.cn; liranyang666@buaa.edu.cn

Junjun Pan
School of Computer Science and Engineering
Beihang University
Email: pan_junjun@buaa.edu.cn
