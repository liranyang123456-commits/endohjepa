# Cover Letter — Medical Image Analysis

Dear Editors and Reviewers,

We submit our manuscript, **“Endo-HJEPA: Hierarchical Latent Prediction for
Cross-Orifice Endoscopic Video with Audited SE(3)-Conditioned Evaluation,”**
for consideration in *Medical Image Analysis*.

Endoscopic AI is predominantly reactive. We study offline prediction of scene
representations and the association between measured camera motion and future
representations, without claiming clinical warning, calibrated collision safety
or robot control. Endo-HJEPA combines a frozen V-JEPA 2 ViT-L encoder across
laparoscopy, gastrointestinal endoscopy and bronchoscopy with residual latent
forecasting and a separately audited SE(3)-conditioned branch.

On a video-level non-overlap offline representation protocol over 19 datasets
and 1,707 sequences, forecast cosine is 0.978 versus 0.916 for persistence,
0.974 for GRU and 0.971 for Mamba. A strictly past-only audit remains above
persistence (0.9578 versus 0.9102). The audited SCARED-subset
action-preference result is 85.2% (n=958); it is explicitly reported as
audit-selected rather than an independent confirmation test. In a one-shot,
oracle-goal latent-retrieval proxy over 200 overlapping windows from four
held-out SCARED sequences, CEM-derived retrieval wins 60.0% of windows and
reduces retrieved-pose translation error by 33.8% versus persistence; this is
not an executed navigation result.
External C3VD action preference is 58.3% across ten usable trajectories
(n=798), below the prespecified gate, and near-wall risk does not generalise
across cases.

The paper's contribution is therefore an executable audit and reproducible
protocol for latent prediction and SE(3)-conditioned offline evaluation. It
documents temporal-context, pose-convention and matched-negative checks, and
reports negative results rather than extending them into unsupported control or
clinical claims. Code, metric records and figure generators are released at
https://github.com/liranyang123456-commits/endohjepa.

This manuscript is original, has not been published previously and is not under
consideration elsewhere. All authors have approved the submission and declare
no competing interests. The ION component was a retrospective secondary
analysis of fully de-identified intra-operative bronchoscopy videos and CT
volumes supplied by Henan Provincial People's Hospital through collaborating
clinician Nan Wei, M.D. The data were de-identified by the hospital in
accordance with its data-handling requirements before transfer to the research
team; the work involved no participant recruitment, clinical intervention, or
direct contact with patients, and the authors had no access to identifiable
personal information or other participant-level records. The use of these
de-identified data was conducted in accordance with the hospital's
requirements.

Sincerely,

Ranyang Li (corresponding author), on behalf of all authors
