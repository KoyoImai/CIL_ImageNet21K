
import os
import numpy as np



import torch
import torchvision.datasets as datasets
import torch.utils.data as data

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

from utils import seed_everything




def encode_filename(fn, max_len=200):
    assert len(
        fn
    ) < max_len, f"Filename is too long. Specified max length is {max_len}"
    fn = fn + '\n' + ' ' * (max_len - len(fn))
    fn = np.fromstring(fn, dtype=np.uint8)
    fn = torch.ByteTensor(fn)
    return fn


def decode_filename(fn):
    fn = fn.cpu().numpy().astype(np.uint8)
    fn = fn.tostring().decode('utf-8')
    fn = fn.split('\n')[0]
    return fn




# ==============================================================
# ImageNet21K データセットの定義
# ==============================================================
class ImageNet21K_ER(data.Dataset):

    def __init__(self, cfg, target_task=None, transforms=None, train=True, replay=False):

        data.Dataset.__init__(self)
        
        self.train = train
        self.replay = replay

        
        # ==============================================================
        # seed値の固定
        # ==============================================================
        seed_everything(seed=cfg.seed)
        

        # ==============================================================
        # filelist.txt までのパスを設定
        # ==============================================================
        self.filelist_dir = cfg.dataset.filelist
        if self.replay:
            if self.train:
                self.current_path = os.path.join(self.filelist_dir, f"task_{target_task:03d}_train.txt")
                self.replay_path = os.path.join(cfg.log.mem_path, f"task_{target_task:03d}_replay.txt")
            else:
                self.current_path = os.path.join(self.filelist_dir, f"task_{target_task:03d}_val.txt")
                self.replay_path = os.path.join(cfg.log.mem_path, f"task_{target_task:03d}_replay.txt")
        else:
            if self.train:
                self.current_path = os.path.join(self.filelist_dir, f"task_{target_task:03d}_train.txt")
                self.replay_path = None
            else:
                self.current_path = os.path.join(self.filelist_dir, f"task_{target_task:03d}_val.txt")
                self.replay_path = None

        
        
        # ==============================================================
        # 特定タスク用のfilelistからパスを読み込む
        # ==============================================================
        with open(self.current_path, 'r') as f:
            all_current_files = f.read().splitlines()      # 使用するデータまでの全てのパスとラベル
        
        if self.replay:
            with open(self.replay_path, 'r') as f:
                all_replay_files = f.read().splitlines()   # リプレイデータまでの全てのパスとラベル
        else:
            all_replay_files = []


        # ==============================================================
        # 現在タスクとリプレイ両方における画像データまでのパスとそのラベルを格納
        # （self.~files）
        # ==============================================================
        self.all_current_files = all_current_files
        self.all_replay_files = all_replay_files


        # ==============================================================
        # 各データのファイルパスのみを獲得
        # （self.~filenames）
        # ==============================================================
        all_current_filenames = [fn.split(" ")[0] for fn in all_current_files]
        if all_replay_files is not []:
            all_replay_filenames = [fn.split(" ")[0] for fn in all_replay_files]
        else:
            all_replay_filenames = []


        # ==============================================================
        # 各データに対応したラベルのみを獲得
        # （self.~labels）
        # ==============================================================
        all_current_labels = [int(fn.split(" ")[-1]) for fn in all_current_files]
        if all_replay_files is not None:
            all_replay_labels = [int(fn.split(" ")[-1]) for fn in all_replay_files]
        else:
            all_replay_labels = []


        # ==============================================================
        # 現在タスクとリプレイのデータとラベルをまとめる
        # ==============================================================
        self.all_files = all_current_files + all_replay_files
        self.all_filenames = all_current_filenames + all_replay_filenames
        self.all_labels = all_current_labels + all_replay_labels



        self.start_replay_idx = len(all_current_files)  # 境界の記録



        # ==============================================================
        # 各データまでの実際のパスをリストに格納
        # ==============================================================
        self.filenames = torch.stack([encode_filename(fn) for fn in self.all_filenames])
        self.labels = torch.tensor(self.all_labels)


        # print("self.filenames.shape: ", self.filenames.shape)    # self.filenames.shape:  torch.Size([78414, 201])

        self.transforms = transforms

    def __getitem__(self, index):
        
        MAX_TRIES = 50
        for i in range(MAX_TRIES):
            try:
                fname = decode_filename(self.filenames[index])
                image = datasets.folder.pil_loader(fname)
                label = self.labels[index]
                break
            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueError(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row. Check {fname}'
                    )
                print(f'Failed to load. {fname}')
                index = np.random.randint(len(self))


        # データ拡張を実行
        if self.transforms is not None:
            image = self.transforms(image)
        # print("image.shape: ", image.shape)

        # meta情報を格納
        meta = {}
        meta["files"] = self.all_files[index]

        return image, label, index, meta

    

    def __len__(self):

        return self.filenames.shape[0]
     



