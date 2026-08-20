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

In a strictly past-only run, mean forecast cosine over steps 1--4 is 0.9578
versus 0.9102 for matched persistence. On a separate 750-clip validation cache
encoded bidirectionally within each clip after 6,000 training clips, the
corresponding values are 0.978 for the forecaster, 0.916 for persistence,
0.974 for GRU and 0.971 for a custom Mamba-inspired gated recurrence. On 958 overlapping
windows from four SCARED audit sequences, forecasts conditioned on recorded
actions have lower error than forecasts conditioned on a no-fixed-point batch
permutation in 88.5% of windows; a distinct
same-sequence bank yields 92.2% pair wins and 66.4% all-negative wins. These
results are audit-selected rather than an independent case-level test.
Near-wall risk does not generalise after correcting label timing (AUC 0.523).

The paper contributes a multi-domain latent forecaster and an executable,
reproducible audit for SE(3)-conditioned association. It documents
temporal-context, pose-convention and negative-action checks, and retains
negative results rather than extending them into unsupported control or
clinical claims. Code, metric records and figure generators are released at
https://github.com/liranyang123456-commits/endohjepa.

This manuscript is original, has not been published previously and is not under
consideration elsewhere. All authors have approved the submission and declare
no competing interests. The private clinical component is a retrospective
secondary analysis of de-identified in-house laparoscopic and ION bronchoscopy
videos, provided by Dr. Nan Wei after the responsible hospital department's
institutional review process. Associated CT volumes were available but were
not used in the reported analyses. Data access and use were reviewed and
authorised through the institutional data-governance process of Henan
Provincial People's Hospital. The authors conducted no new animal experiments,
prospective enrolment or research-directed intervention, and no identifiable
participant information was used. The study complies with the Declaration of
Helsinki. The formal ethics approval or exemption identifier, decision date
and consent-waiver determination must be supplied by the responsible
institution before submission.

Sincerely,

Ranyang Li (corresponding author), on behalf of all authors
