# Cover Letter — Medical Image Analysis

Dear Editors and Reviewers,

We submit our manuscript, **“Endo-HJEPA: Multi-Domain Latent Forecasting with
Audited SE(3)-Conditioned Dynamics for Endoscopic Video,”**
for consideration in *Medical Image Analysis*.

We study two questions that are often conflated in endoscopic world models:
offline prediction of future scene representations and sensitivity of those
predictions to measured camera motion. Endo-HJEPA combines a frozen V-JEPA 2
ViT-L encoder across laparoscopic, gastrointestinal and bronchoscopic video
with residual latent forecasting and a separately audited
SE(3)-conditioned branch.

On a common 750-clip validation set after 6,000 training clips, mean forecast
cosine over steps 1--4 is 0.978 versus 0.916 for persistence, 0.974 for GRU
and 0.971 for Mamba. A separate strictly past-only audit remains above its
matched persistence baseline (0.9578 versus 0.9102). On 958 overlapping
windows from four SCARED audit sequences, real actions outperform a
no-fixed-point batch permutation in 87.0% of windows; a distinct
same-sequence bank yields 91.3% pair wins and 66.5% all-negative wins. These
results are audit-selected rather than an independent case-level test.
Near-wall risk does not generalise after correcting label timing (AUC 0.523).

The paper contributes a multi-domain latent forecaster and an executable,
reproducible audit for SE(3)-conditioned association. It documents
temporal-context, pose-convention and matched-negative checks, and retains
negative results rather than extending them into unsupported control or
clinical claims. Code, metric records and figure generators are released at
https://github.com/liranyang123456-commits/endohjepa.

This manuscript is original, has not been published previously and is not under
consideration elsewhere. All authors have approved the submission and declare
no competing interests. The ION component is a retrospective secondary analysis
of de-identified CT volumes and intra-operative bronchoscopy videos, provided
by Dr. Nan Wei after the responsible hospital department's strict institutional
review process. Data access and use were reviewed and authorised through the
institutional data-governance process of Henan Provincial People's Hospital.
No animal experiments, prospective enrolment, research-directed intervention
or identifiable participant information were involved, and the study complies
with the Declaration of Helsinki.

Sincerely,

Ranyang Li (corresponding author), on behalf of all authors
