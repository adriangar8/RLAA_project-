"""
Behavioral Cloning (BC) trainer.

BC = supervised learning on (obs, action) pairs from expert.
Loss: cross-entropy between π_θ(s) and expert action a*.

Key limitation (motivates RL fine-tuning):
  BC minimizes E_{s ~ d_expert}[L(π_θ(s), a*)]
  But at test time, the agent visits d_{π_θ} which diverges from d_expert.
  Errors compound: mistake bound grows as O(T² ε) [Ross & Bagnell, 2010].
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass
from pathlib import Path

from src.policy import PolicyNetwork
from src.expert import DemonstrationDataset


class DemoTorchDataset(Dataset):
    def __init__(self, demo: DemonstrationDataset):
        self.obs = torch.FloatTensor(demo.observations)
        self.actions = torch.LongTensor(demo.actions)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


@dataclass
class BCConfig:
    lr: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 50
    val_fraction: float = 0.1
    weight_decay: float = 1e-4
    patience: int = 10             # early stopping
    save_dir: str = "checkpoints/bc"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BehavioralCloning:
    def __init__(self, policy: PolicyNetwork, cfg: BCConfig):
        self.policy = policy.to(cfg.device)
        self.cfg = cfg
        self.optimizer = optim.Adam(
            policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    def train(self, demo: DemonstrationDataset) -> dict:
        """
        Train policy via BC on demonstrations.
        Returns training history dict with train/val loss per epoch.
        """
        cfg = self.cfg
        dataset = DemoTorchDataset(demo)

        val_size = int(len(dataset) * cfg.val_fraction)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg.n_epochs):
            # --- Train ---
            self.policy.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for obs, actions in train_loader:
                obs = obs.to(cfg.device)
                actions = actions.to(cfg.device)

                logits = self.policy(obs)
                loss = self.loss_fn(logits, actions)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * len(obs)
                train_correct += (logits.argmax(1) == actions).sum().item()
                train_total += len(obs)

            # --- Validate ---
            val_loss, val_correct, val_total = self._evaluate(val_loader)

            train_loss /= train_total
            val_loss /= val_total
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{cfg.n_epochs} | "
                      f"train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
                      f"val loss: {val_loss:.4f} acc: {val_acc:.3f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save("best_bc.pt")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best
        self.load("best_bc.pt")
        print(f"\nBC training done. Best val loss: {best_val_loss:.4f}")
        return history

    def _evaluate(self, loader: DataLoader):
        self.policy.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for obs, actions in loader:
                obs = obs.to(self.cfg.device)
                actions = actions.to(self.cfg.device)
                logits = self.policy(obs)
                loss = self.loss_fn(logits, actions)
                total_loss += loss.item() * len(obs)
                correct += (logits.argmax(1) == actions).sum().item()
                total += len(obs)
        return total_loss, correct, total

    def save(self, name: str):
        path = os.path.join(self.cfg.save_dir, name)
        torch.save(self.policy.state_dict(), path)

    def load(self, name: str):
        path = os.path.join(self.cfg.save_dir, name)
        self.policy.load_state_dict(torch.load(path, map_location=self.cfg.device))
