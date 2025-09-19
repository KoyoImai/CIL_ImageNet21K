

import math
import torch




class CosineAnnealingWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, min_lr=0.0, last_epoch=-1):

        """
        Args:
            optimizer (torch.optim.Optimizer): PyTorch optimizer
            warmup_epochs (int): Warmup期間（エポック単位）
            total_epochs (int): 総エポック数
            steps_per_epoch (int): 1エポックあたりのステップ数（len(dataloader)）
            min_lr (float): CosineDecay後の最小学習率
            last_epoch (int): 初期値（通常は-1）
        """
        
        self.warmup_steps = warmup_epochs * steps_per_epoch      # warmupをステップ数に変換
        self.total_steps = total_epochs * steps_per_epoch        # 総学習回数をステップ数に変換
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)
        

    def get_lr(self):

        step = self.last_epoch + 1  # 現在のステップ番号

        # --- 1. Warmup: 線形増加 ---
        if step <= self.warmup_steps:
            return [
                base_lr * step / self.warmup_steps
                for base_lr in self.base_lrs
            ]

        # --- 2. Cosine decay ---
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return [
            self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]









