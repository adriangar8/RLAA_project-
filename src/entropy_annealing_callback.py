from stable_baselines3.common.callbacks import BaseCallback


class EntropyAnnealingCallback(BaseCallback):
    """
    Linearly anneals PPO entropy coefficient during training.

    ent_coef(t) = start_ent * (1 - progress) + end_ent * progress
    where progress ∈ [0,1]
    """

    def __init__(self, start_ent: float = 0.02, end_ent: float = 0.001, verbose=0):
        super().__init__(verbose)
        self.start_ent = start_ent
        self.end_ent = end_ent

    def _on_training_start(self) -> None:
        self.total_timesteps = self.model._total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps

        new_entropy = self.start_ent * (1 - progress) + self.end_ent * progress

        # Update PPO entropy coefficient
        self.model.ent_coef = new_entropy

        if self.verbose > 0 and self.num_timesteps % 50000 == 0:
            print(f"Entropy coef updated → {new_entropy:.6f}")

        return True
    