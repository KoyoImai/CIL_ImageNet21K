


import torch





class MultiStepWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, milestones, steps_per_epoch, gamma=0.1, last_epoch=-1):

        
        
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs (int): Warmup期間（エポック単位）
            milestones (list[int]): 学習率を減衰させるエポックリスト
            steps_per_epoch (int): 1エポックあたりの更新ステップ数
            gamma (float): 減衰率
        """
        
        self.warmup_epochs = warmup_epochs
        self.milestones = sorted(milestones)
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        

    def get_lr(self):
        
        # 現在のエポックとステップを計算
        current_step = self.last_epoch + 1  # _LRScheduler内部はstepベース
        current_epoch = current_step / self.steps_per_epoch

        if current_epoch <= self.warmup_epochs:
            
            # --- Warmup期間: 線形にbase_lrまで増加 ---
            return [
                base_lr * current_epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        
        else:
            
            # --- Warmup終了後: MultiStepDecay ---
            decay_factor = self.gamma ** sum(current_epoch >= m for m in self.milestones)
            return [
                base_lr * decay_factor
                for base_lr in self.base_lrs
            ]
        














