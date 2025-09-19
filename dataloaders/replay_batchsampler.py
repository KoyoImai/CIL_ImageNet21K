

import torch
import torch.distributed as dist
import numpy as np
import math
from torch.utils.data import Sampler



# ================================================
# replay_indices を各プロセスで共有する BatchSampler
# ================================================
class MixedReplayBatchSampler(Sampler):

    """
    現在タスクとリプレイデータを一定比率で混ぜるBatchSampler（DDP対応）
    
    Args:
        dataset: ImageNet21K_ER
        batch_size: 1GPUあたりのミニバッチサイズ
        current_ratio: 現在タスクの割合（0.0～1.0）
        drop_last: 最後の不完全なバッチを捨てるか
        seed: シャッフル用の初期シード
    """
    
    def __init__(self, dataset, batch_size, cfg=None, current_ratio=0.7, drop_last=True, seed=42):

        # ==== 初期化 ====
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_ratio = current_ratio
        self.drop_last = drop_last
        self.seed = seed


        # ==== DDP情報 ====
        if dist.is_available() and dist.is_initialized():
            self.rank = cfg.ddp.local_rank
            self.world_size = cfg.ddp.world_size 
        else:
            self.rank = 0
            self.world_size = 1

        
        # ==== 現在タスクとリプレイ範囲 ====
        self.num_current = dataset.start_replay_idx
        self.num_replay = len(dataset) - dataset.start_replay_idx


        # 1rank あたりの現在タスク数
        self.num_current_per_rank = math.ceil(self.num_current / self.world_size)
        self.total_current_size = self.num_current_per_rank * self.world_size


        # 1バッチあたりの各種サイズ
        self.n_cur = int(round(self.current_ratio * self.batch_size))
        self.n_cur = min(self.n_cur, self.batch_size)
        self.n_rep = self.batch_size - self.n_cur


        # エポックあたりの総バッチ数は現在タスク基準（リプレイサンプルは総バッチ数に含まない）
        self.num_batches = self.num_current_per_rank // self.n_cur
        if not self.drop_last and self.num_current_per_rank % self.n_cur != 0:
            self.num_batches += 1


    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed)

        # ====== 現在タスク ======
        current_indices = torch.randperm(self.num_current, generator=g).tolist()

        # padして各rankが同数になるように調整
        if len(current_indices) < self.total_current_size:
            current_indices += (
                current_indices * ((self.total_current_size - len(current_indices)) // len(current_indices) + 1)
            )[: self.total_current_size - len(current_indices)]
        
        # print("len(current_indices): ", len(current_indices))
        # assert False

        # rankごとの分割
        start = self.rank * self.num_current_per_rank
        end = start + self.num_current_per_rank
        current_indices_rank = current_indices[start:end]
        # print("self.rank: {}, start: {}, end: {}".format(self.rank, start, end))


        # ====== リプレイ ======
        if self.num_replay > 0 and self.n_rep > 0:
            replay_indices_all = np.arange(self.dataset.start_replay_idx, len(self.dataset))
        else:
            replay_indices_all = None
        

        # ====== バッチ生成 ======
        cur_ptr = 0
        for _ in range(self.num_batches):
            batch = []

            # --- 現在タスク部分 ---
            cur_batch = current_indices_rank[cur_ptr:cur_ptr + self.n_cur]
            cur_ptr += self.n_cur
            batch.extend(cur_batch)

            # --- リプレイ部分 ---
            if self.n_rep > 0 and replay_indices_all is not None:
                rep_sample = np.random.choice(replay_indices_all, self.n_rep, replace=True)
                batch.extend(rep_sample.tolist())

            yield batch

    
    def set_epoch(self, epoch):

        self.seed = epoch


    def __len__(self):
        return self.num_batches




