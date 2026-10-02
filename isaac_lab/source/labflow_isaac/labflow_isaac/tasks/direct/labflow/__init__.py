import gymnasium as gym

TASK_ID = "Isaac-LabFlow-Jetbot-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point=(
        "labflow_isaac.tasks.direct.labflow.labflow_env:"
        "LabFlowIsaacEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "labflow_isaac.tasks.direct.labflow.labflow_env:"
            "LabFlowIsaacEnvCfg"
        )
    },
)

__all__ = ["TASK_ID"]
