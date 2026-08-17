"""Lung nodule ablation planning.

Pipeline: parse clinical records -> load CT -> segment lung/nodule/vessels ->
bioheat ablation-zone model -> optimise needle path + per-burn power/time for full
tumor coverage with a safety margin while sparing critical structures -> evaluate.

Learning / simulation loop
--------------------------
``trajectory_schema``  — typed ``(θ, M_pre, {(p,T,t)}, M_post, outcome)`` records
``sim_env``            — Gym-style sequential ablation environment
``dataset`` / ``policy`` / ``safety_gate`` / ``preference`` / ``train_eval``
``segment3d``          — Cohort A: ION CT → lung/nodule/vessel 3D masks
``followup_masks``     — Cohort B: unzip vue + per-timepoint zone masks
``patient_sim``        — 术前3D → burns → 合成术后区
``run_upgrade``        — one-shot A+B+sim

    PYTHONPATH=src python -m endoworld.ablation.run_upgrade --limit-a 2 --limit-b 4
"""

from endoworld.ablation.trajectory_schema import (  # noqa: F401
    AblationTrajectory,
    BurnStep,
    DeviceParams,
    LesionGeometry,
    OutcomeLabel,
    load_trajectory,
    save_trajectory,
    plan_to_trajectory,
)
from endoworld.ablation.sim_env import (  # noqa: F401
    AblationSimEnv,
    AblationAction,
    EnvConfig,
    make_env_from_axes,
    rollout,
)
from endoworld.ablation.policy import AblationPolicy, load_policy  # noqa: F401
from endoworld.ablation.safety_gate import GateConfig, gate_rollout  # noqa: F401
