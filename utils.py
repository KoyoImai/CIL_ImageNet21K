
import os
import csv

import random
import numpy as np
import torch



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


        

def seed_everything(seed):
    # Python 内部のハッシュシードを固定（辞書等の再現性に寄与）
    # os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Python 標準の乱数生成器のシード固定
    random.seed(seed)
    
    # NumPy の乱数生成器のシード固定
    np.random.seed(seed)
    
    # PyTorch のシード固定
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # マルチGPU対応の場合
    # Deterministic モードの有効化（PyTorch の一部非決定的な処理の回避）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    

    

# csvファイルに値を書き込む
def write_csv(value, path, file_name, task, epoch):
    # ファイルパスを生成
    file_path = f"{path}/{file_name}.csv"

    # ファイルが存在しなければ新規作成、かつヘッダー行を記入する
    # value がリストの場合は、ヘッダーの値部分は要素数に合わせて "value_1", "value_2", ... とする例
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行を定義（必要に応じて適宜変更）
            if isinstance(value, list):
                header = ["task"] + ["epoch"] + [f"task_{i+1}" for i in range(len(value))]
            else:
                header = ["task", "epoch", "value"]
            writer.writerow(header)

    # CSV に実際のデータを追加記録する
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if isinstance(value, list):
            row = [task] + [epoch] + value
        else:
            row = [task, epoch, value]
        writer.writerow(row)


def save_replay_indices_to_txt(replay_indices, save_path):
    """
    replay_indices の内容を1行ずつテキストファイルに保存する関数
    """
    with open(save_path, "w") as f:
        for item in replay_indices:
            f.write(f"{item}\n")  # 各要素を1行に書き込む




def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...'+save_file)
    if opt.method in ["cclis-pcgrad"]:
        state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer._optim.state_dict(),
        'epoch': epoch,
    }
    else:
        state = {
            'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
    torch.save(state, save_file)
    del state




# 学習途中のパラメータなどを読み込む
def load_checkpoint(cfg, model, model2, optimizer, scheduler, filepath):
    checkpoint = torch.load(filepath, map_location='cuda:{}'.format(cfg.ddp.local_rank))

    model.module.load_state_dict(checkpoint['model'])
    model2.module.load_state_dict(checkpoint['model2'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    replay_indices = checkpoint['replay_indices']
    target_task = checkpoint['target_task']
    start_epoch = checkpoint['epoch'] + 1  # 次のエポックから開始

    # RNGの再現性確保
    torch.set_rng_state(checkpoint['rng_state']['torch'])
    torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
    np.random.set_state(checkpoint['rng_state']['numpy'])
    random.setstate(checkpoint['rng_state']['random'])

    print(f"[INFO] Checkpoint loaded from {filepath}")
    return replay_indices, target_task, start_epoch




# 学習途中のパラメータを保存する関数
def save_checkpoint(cfg, model, model2, optimizer, scheduler, replay_indices, target_task, epoch, filepath):
    if cfg.ddp.local_rank == 0:  # rank0のみが保存
        state = {
            'model': model.module.state_dict(),  # DDPでは .module が必要
            'model2': model2.module.state_dict(),  # DDPでは .module が必要
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'replay_indices': replay_indices,
            'target_task': target_task,
            'epoch': epoch,
            'rng_state': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all(),
                'numpy': np.random.get_state(),
                'random': random.getstate()
            }
        }
        torch.save(state, filepath)
        print(f"[INFO] Checkpoint saved to {filepath}")
