
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler, BatchSampler


import torch
import torch.distributed as dist
import pickle

from dataloaders.replay_batchsampler import MixedReplayBatchSampler

# er
from dataloaders.dataset_er import ImageNet21K_ER









# ==============================================================
# DDP使用のための共通へルパ
# ==============================================================
def build_laoder_ddp(cfg, dataset, batch_size, num_workers, is_train=True):


    sampler = None
    shuffle = is_train
    drop_last = is_train

    if cfg.ddp.use_ddp:
        sampler  = DistributedSampler(dataset, shuffle=is_train, drop_last=is_train)
        shuffle  = False     # Samplerがshuffleを担当
        drop_last = is_train # 明示
    

    loader = DataLoader(
        dataset,
        batch_size=batch_size,          # ← per-GPU（総バッチ=この値×world_size）
        shuffle=shuffle,                # ← DDP時は False
        sampler=sampler,                # ← DDP時のみ設定
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )

    return loader




# ==============================================================
# データセットの作成　〜　データローダーの作成までを実行
# ==============================================================
def set_loader(cfg, model, replay_indices):

    if cfg.dataset.type == 'imagenet21k':
        mean=(0.430, 0.411, 0.296)
        std=(0.213, 0.156, 0.143)

    normalize = transforms.Normalize(mean=mean, std=std)


    if cfg.method.name in ["er"]:

        train_transforms = transforms.Compose([
            transforms.Resize(size=(cfg.dataset.size, cfg.dataset.size)),
            transforms.RandomResizedCrop(size=cfg.dataset.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(cfg.dataset.size, cfg.dataset.size)),
            transforms.RandomResizedCrop(size=cfg.dataset.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


        # ==============================================================
        # データセットの作成
        # ==============================================================     
        if cfg.continual.target_task != 0:
            train_dataset = ImageNet21K_ER(cfg, transforms=train_transforms, target_task=cfg.continual.target_task, train=True, replay=True)
        else:
            train_dataset = ImageNet21K_ER(cfg, transforms=train_transforms, target_task=cfg.continual.target_task, train=True, replay=False)
        
        # print("len(train_dataset): ", len(train_dataset))   # len(train_dataset):  78414
        
        # ==============================================================
        # DDP 用バッチサンプラーの作成
        # 現在タスクはプロセス毎に分割，リプレイデータは全プロセスで共通
        # ==============================================================
        if train_dataset.replay:
            current_ratio = cfg.optimizer.train.current_ratio
        else:
            current_ratio = 1.0
        
        batch_sampler = MixedReplayBatchSampler(
            dataset=train_dataset,
            batch_size=cfg.optimizer.train.batch_size,
            cfg=cfg,
            current_ratio=current_ratio,  # 例: 0.7
            drop_last=True,
            seed=cfg.seed
        )


        # ==============================================================
        # データローダーの作成
        # ==============================================================
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.workers,
            pin_memory=True
        )
        print("len(train_loader): ", len(train_loader))


        # for (images, labels, index, meta) in train_loader:

        #     print("images.shape: ", images.shape)
        #     print("labels.shape: ", labels.shape)

        #     assert False


        return train_loader



# ==========================
# 検証用データローダーメインの作成
# ==========================
def set_loader_eval(cfg, model, replay_indices, method_tools):

    if cfg.dataset.type == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif cfg.dataset.type == 'cifar100':       # scaleから
        # mean = (0.5071, 0.4867, 0.4408)
        # std = (0.2675, 0.2565, 0.2761)
        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
    elif cfg.dataset.type == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif cfg.dataset.type == 'path':
        mean = eval(cfg.mean)
        std = eval(cfg.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(cfg.dataset.type))

    normalize = transforms.Normalize(mean=mean, std=std)

    if cfg.dataset.type == "cifar10":
        val_loader = set_valloader_co2l_cifar10(opt=cfg, normalize=normalize)
        linear_loader = set_linearloader_co2l_cifar10(opt=cfg, normalize=normalize, replay_indices=replay_indices)
        ncm_loader, _ = set_ncmloader_er_cifar10(opt=cfg, normalize=normalize, replay_indices=replay_indices)
        taskil_loaders = set_taskil_valloader_er_cifar10(opt=cfg, normalize=normalize)
        knn_loaders = set_taskil_valloader_er_cifar10(opt=cfg, normalize=normalize, train=True)
    elif cfg.dataset.type == "cifar100":
        val_loader = set_valloader_co2l_cifar100(opt=cfg, normalize=normalize)
        linear_loader = set_linearloader_co2l_cifar100(opt=cfg, normalize=normalize, replay_indices=replay_indices)
        ncm_loader, _ = set_ncmloader_er_cifar100(opt=cfg, normalize=normalize, replay_indices=replay_indices)
        taskil_loaders = set_taskil_valloader_er_cifar100(opt=cfg, normalize=normalize)
        knn_loaders = set_taskil_valloader_er_cifar100(opt=cfg, normalize=normalize, train=True)
    elif cfg.dataset.type == 'tiny-imagenet':
        val_loader = set_valloader_co2l_tinyimagenet(opt=cfg, normalize=normalize)
        linear_loader = set_linearloader_co2l_tinyimagenet(opt=cfg, normalize=normalize, replay_indices=replay_indices)
        ncm_loader, _ = set_ncmloader_er_tinyimagenet(opt=cfg, normalize=normalize, replay_indices=replay_indices)
        taskil_loaders = set_taskil_valloader_er_tinyimagenet(opt=cfg, normalize=normalize)
        knn_loaders = set_taskil_valloader_er_tinyimagenet(opt=cfg, normalize=normalize, train=True)
    


    dataloaders = {"val": val_loader, "linear": linear_loader, "ncm": ncm_loader, "taskil": taskil_loaders, "knn": knn_loaders}

    return dataloaders


# ====================================================
# tiny-imagenet専用，検証用データローダーメインの作成
# ====================================================
def set_loader_eval4timnet(cfg, model, replay_indices, method_tools):

    if cfg.dataset.type == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif cfg.dataset.type == 'path':
        mean = eval(cfg.mean)
        std = eval(cfg.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(cfg.dataset.type))

    normalize = transforms.Normalize(mean=mean, std=std)


    if cfg.dataset.type == 'tiny-imagenet':
        val_loader = set_valloader_co2l_tinyimagenet(opt=cfg, normalize=normalize)
        linear_loader = set_linearloader_co2l_tinyimagenet(opt=cfg, normalize=normalize, replay_indices=replay_indices)
        taskil_loaders = set_taskil_valloader_er_tinyimagenet(opt=cfg, normalize=normalize)
        # taskil_loaders = None


    dataloaders = {"val": val_loader, "linear": linear_loader, "taskil": taskil_loaders}

    return dataloaders




# ==============================================================
# バッファの作成
# （要改良．filelistを参照して保存データを選択できるように改良する）
# ==============================================================
def set_buffer(cfg, model, prev_indices=None):

    # replay_inidcesの初期化
    replay_indices = None

    
    if cfg.method.name in ["er"]:

        from dataloaders.buffer_er import set_replay_samples_ring

        if cfg.continual.mem_type == "ring":
            if cfg.ddp.local_rank == 0:
                replay_indices = set_replay_samples_ring(cfg, model, prev_indices=prev_indices)

                # Pythonオブジェクト（リスト）をバイト列に変換
                data_bytes = pickle.dumps(replay_indices)
                data_tensor = torch.ByteTensor(list(data_bytes)).to('cuda')

                # データサイズをブロードキャストするためにtensorで共有
                size_tensor = torch.tensor([data_tensor.size(0)], dtype=torch.long, device='cuda')
            else:
                # rank 0から送られるデータサイズを受信するためのtensor
                size_tensor = torch.tensor([0], dtype=torch.long, device='cuda')

            # ====== 全プロセスでサイズ情報を同期 ======
            if cfg.ddp.use_ddp:
                dist.broadcast(size_tensor, src=0)

            # rank 0以外では受信用のbufferを確保
            if cfg.ddp.local_rank != 0:
                data_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device='cuda')

            # ====== バイト列データを同期 ======
            if cfg.ddp.use_ddp:
                dist.broadcast(data_tensor, src=0)

            # rank 0以外は受信データをデシリアライズ
            if cfg.ddp.local_rank != 0:
                data_bytes = bytes(data_tensor.tolist())
                replay_indices = pickle.loads(data_bytes)

        else:
            assert False
        
        print("len(replay_indices): ", len(replay_indices))

    
    return replay_indices