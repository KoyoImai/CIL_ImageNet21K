


import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist






# ===========================================
# replay_indices を各プロセスで共有する関数
# ===========================================
def bcast_from_main_pyobj(pyobj, src=0):
    
    # rank0: pyobj、他rank: None を渡し、全rankで同じオブジェクトを得る
    lst = [pyobj] if (not dist.is_available() or not dist.is_initialized() or dist.get_rank()==src) else [None]
    
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(lst, src=src)
    
    return lst[0]


def sync_dict_across_ranks(obj_dict, src=0):
    """
    Pythonのdictをrank0から全rankにbroadcastする
    """
    if not dist.is_initialized():
        return obj_dict
    container = [obj_dict if dist.get_rank() == src else None]
    dist.broadcast_object_list(container, src=src)
    return container[0]






# ===========================================================
# replay_indices を全てのプロセスで共有するための Batch Sampler
# ===========================================================
class ReplayBatchSampler(Sampler):
    def __init__(self, current_indices, replay_indices,
                 batch_size_new, batch_size_replay, shuffle=True, drop_last=True):
        """
        current_indices: 新タスクデータのインデックスリスト
        replay_indices:  リプレイデータのインデックスリスト
        batch_size_new:  新タスク部分の1ミニバッチサイズ (per-GPU)
        batch_size_replay: リプレイ部分の1ミニバッチサイズ (全rank共通)
        """
        self.current_indices = current_indices
        self.replay_indices = replay_indices
        self.batch_size_new = batch_size_new
        self.batch_size_replay = batch_size_replay
        self.shuffle = shuffle
        self.drop_last = drop_last

        # DDP情報
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # ===== 新タスクデータ: rankごとに分割 =====
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch if hasattr(self, "epoch") else 0)
            current_indices = torch.randperm(len(self.current_indices), generator=g).tolist()
        else:
            current_indices = list(self.current_indices)

        # 各rankに分割
        total_per_rank = int(math.ceil(len(current_indices) / self.world_size))
        start = self.rank * total_per_rank
        end = min(start + total_per_rank, len(current_indices))
        current_rank_indices = current_indices[start:end]

        # ===== リプレイデータ: 全rank共通 =====
        replay_indices = list(self.replay_indices)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch if hasattr(self, "epoch") else 0)
            replay_indices = torch.tensor(replay_indices)[torch.randperm(len(replay_indices), generator=g)].tolist()

        # ===== バッチを生成 =====
        num_batches = len(current_rank_indices) // self.batch_size_new
        for i in range(num_batches):
            batch_new = current_rank_indices[i*self.batch_size_new : (i+1)*self.batch_size_new]

            # リプレイは全rankで同じサンプルを取る
            replay_start = (i * self.batch_size_replay) % len(replay_indices)
            batch_replay = replay_indices[replay_start : replay_start + self.batch_size_replay]

            yield batch_new + batch_replay

    def __len__(self):
        return len(self.current_indices) // (self.batch_size_new * self.world_size)











