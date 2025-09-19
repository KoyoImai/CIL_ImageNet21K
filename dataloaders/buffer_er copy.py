import random
import math
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

import torch.distributed as dist

from dataloaders.dataset_er import ImageNet21K_ER
from dataloaders.replay_utils import bcast_from_main_pyobj



def set_replay_samples_reservoir(cfg, model, prev_indices=None):

    
    if dist.get_rank() != 0:
        # rank0以外は空処理
        total_indices = None
    else:
        # rank0だけが実際に計算
        is_training = model.training
        model.eval()

        class IdxDataset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                return self.indices[idx], self.dataset[idx]

        # データセットの仮作成（ラベルがほしいだけ）
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if cfg.dataset == 'cifar10':
            subset_indices = []
            val_dataset = datasets.CIFAR10(root=cfg.data_folder,
                                            transform=val_transform,
                                            download=True)
            val_targets = np.array(val_dataset.targets)
        elif cfg.dataset == 'cifar100':
            subset_indices = []
            val_dataset = datasets.CIFAR100(root=cfg.data_folder,
                                            transform=val_transform,
                                            download=True)
            val_targets = np.array(val_dataset.targets)
        elif cfg.dataset == 'tiny-imagenet':
            subset_indices = []
            val_dataset = TinyImagenet(root=cfg.data_folder,
                                        transform=val_transform,
                                        download=True)
            val_targets = val_dataset.targets

        else:
            raise ValueError('dataset not supported: {}'.format(cfg.dataset))
        
        # 前回タスクのクラスを獲得
        if prev_indices is None:
            prev_indices = []
            observed_classes = list(range(0, cfg.target_task*cfg.cls_per_task))
        else:
            observed_classes = list(range(max(cfg.target_task-1, 0)*cfg.cls_per_task, (cfg.target_task)*cfg.cls_per_task))

        if len(observed_classes) == 0:
            total_indices = prev_indices
        else:

            # 前回タスクのデータのインデックス獲得
            observed_indices = []
            for tc in observed_classes:
                observed_indices += np.where(val_targets == tc)[0].tolist()
            
            total_indices = prev_indices + observed_indices
            # print("1 total_indices: ", total_indices)
            
            # ランダムにバッファサイズ分だけ取り出す
            random.shuffle(total_indices)
            # print("2 total_indices: ", total_indices)
            # assert False

            total_indices = total_indices[:cfg.mem_size]
    
    # 全rankに配布
    total_indices = bcast_from_main_pyobj(total_indices)

    return total_indices



# ring buffer
def set_replay_samples_ring(cfg, model, prev_indices=None):


    if dist.get_rank() != 0:
        # rank0以外は空処理
        total_indices = None
    else:
        # rank0だけが実際に計算

        is_training = model.training
        model.eval()

        class IdxDataset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                return self.indices[idx], self.dataset[idx]


        # データセットの仮作成
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if cfg.dataset.type == 'imagenet21k':
            subset_indices = []
            train_dataset = ImageNet21K_ER(cfg, transforms=transform, target_task=cfg.continual.target_task-1, train=True, replay=False)
        else:
            raise ValueError('dataset not supported: {}'.format(cfg.dataset.type))
        

        if prev_indices is None:
            prev_indices = []
            # observed_classes = list(range(0, cfg.continual.cls_per_task[0]))
            observed_classes = []
        else:
            target_task = cfg.continual.target_task
            observed_classes = list(range(sum(cfg.continual.cls_per_task[:target_task])))
            print("observed_classes: ", observed_classes)

            assert False


            # 過去タスクのデータに割り当てるバッファのサイズ
            shrink_size = ((cfg.continual.target_task - 1) * cfg.continual.mem_size / cfg.continual.target_task)

            
            # 前回タスクのクラス範囲
            observed_classes = list(range(max(cfg.continual.target_task-1, 0)*cfg.continual.cls_per_task, (cfg.continual.target_task)*cfg.continual.cls_per_task))
        
        print("buffer_er.py observed_classes: ", observed_classes)

        # 確認済みのクラス（前回タスク）がない場合終了
        if len(observed_classes) == 0:
            total_indices = prev_indices
        else:

            # 確認済みクラスのインデックスを獲得
            observed_indices = []
            for tc in observed_classes:
                observed_indices += np.where(val_targets == tc)[0].tolist()


            val_observed_targets = val_targets[observed_indices]
            val_unique_cls = np.unique(val_observed_targets)
            print("val_unique_cls: ", val_unique_cls)


            print("cfg.continual.mem_size: ", cfg.continual.mem_size)
            selected_observed_indices = []
            for c_idx, c in enumerate(val_unique_cls):
                size_for_c_float = ((cfg.continual.mem_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
                print("size_for_c_flaot: ", size_for_c_float)
                p = size_for_c_float -  ((cfg.continual.mem_size - len(prev_indices) - len(selected_observed_indices)) // (len(val_unique_cls) - c_idx))
                if random.random() < p:
                    size_for_c = math.ceil(size_for_c_float)
                else:
                    size_for_c = math.floor(size_for_c_float)
                mask = val_targets[observed_indices] == c
                selected_observed_indices += torch.tensor(observed_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()
            print(np.unique(val_targets[selected_observed_indices], return_counts=True))

            total_indices = prev_indices + selected_observed_indices
            
        model.is_training = is_training
        
    # 全rankに配布
    total_indices = bcast_from_main_pyobj(total_indices)

    rank = dist.get_rank()
    print("target_task: {}, rank: {}. total_indices[:50]: {},".format(cfg.continual.target_task, rank, total_indices[:50]))



    return total_indices