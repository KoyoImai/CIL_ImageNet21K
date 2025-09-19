
import os
import copy
import hydra
import random
import numpy as np


import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim


from utils import seed_everything, save_model, save_replay_indices_to_txt
from models import make_model
from dataloaders import set_buffer, set_loader
from schedulers import make_scheduler
from train import train





def preparation(cfg):

    # データセット毎にタスク数・タスク毎のクラス数を決定
    # （現状不要）

    # 総タスク数
    # （現状不要）

    # モデルの保存，実験記録などの保存先パス
    if cfg.dataset.data_folder is None:
        cfg.dataset.data_folder = '~/data/'
    cfg.log.model_path = f'./logs/{cfg.method.name}/{cfg.log.name}/model/'      # modelの保存先
    cfg.log.explog_path = f'./logs/{cfg.method.name}/{cfg.log.name}/exp_log/'   # 実験記録の保存先
    cfg.log.mem_path = f'./logs/{cfg.method.name}/{cfg.log.name}/mem_log/'      # リプレイバッファ内の保存先
    cfg.log.result_path = f'./logs/{cfg.method.name}/{cfg.log.name}/result/'    # 結果の保存先

    # ディレクトリ作成
    if not os.path.isdir(cfg.log.model_path):
        os.makedirs(cfg.log.model_path)
    if not os.path.isdir(cfg.log.explog_path):
        os.makedirs(cfg.log.explog_path)
    if not os.path.isdir(cfg.log.mem_path):
        os.makedirs(cfg.log.mem_path)
    if not os.path.isdir(cfg.log.result_path):
        os.makedirs(cfg.log.result_path)



def make_setup(cfg):

    if cfg.method.name in ["er"]:

        from models.resnet_er import BackboneResNet

        model = BackboneResNet(name='resnet50', head='linear', feat_dim=128, seed=777, opt=cfg)
        model2 = None
        # print("model: ", model)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.optimizer.train.learning_rate,
                              momentum=cfg.optimizer.train.momentum,
                              weight_decay=cfg.optimizer.train.weight_decay)

    elif cfg.method.name in ["co2l"]:
        
        assert False

    elif cfg.method.name in ["cclis"]:

        assert False

    elif cfg.method.name in ["prco"]:

        assert False
    
    else:

        assert False
    

    if torch.cuda.is_available():
        model = model.cuda()
        if model2 is not None:
            model2 = model2.cuda()
        criterion = criterion.cuda()
    

    return model, model2, criterion, optimizer








@hydra.main(config_path='configs/default/', config_name='default', version_base=None)
def main(cfg):

    # ===========================================
    # シード固定
    # ===========================================
    seed_everything(cfg.seed)


    # logの名前
    cfg.log.name = f"{cfg.log.base}_{cfg.method.name}_{cfg.continual.mem_type}{cfg.continual.mem_size}_{cfg.dataset.type}_seed{cfg.seed}_date{cfg.date}"


    # ===========================================
    # データローダ作成の前処理
    # ===========================================
    preparation(cfg)


    # ===========================================
    # DDP
    # ===========================================
    
    # DDP 使用の環境変数
    local_rank = int(os.environ["LOCAL_RANK"])
    use_ddp = local_rank != -1
    device = torch.device("cuda", local_rank if use_ddp else 0)
    
    cfg.ddp.local_rank = local_rank
    cfg.ddp.use_ddp = use_ddp
    cfg.ddp.world_size = int(os.environ['WORLD_SIZE'])

    # DDP 使用
    if use_ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        print("dist.get_world_size(): ", dist.get_world_size())
    else:
        assert False



    # ===========================================
    # モデル，損失関数，Optimizer の作成
    # ===========================================
    model, model2, criterion, optimizer = make_setup(cfg)


    # バッファ内データのインデックス
    replay_indices = None

    # タスク毎の学習エポック数
    original_epochs = cfg.optimizer.train.epochs



    # ===========================================
    # 学習途中のパラメータがあれば読み込む
    # ===========================================





    # ===========================================
    # 各タスクを順番に学習
    # ===========================================
    for target_task in range(0, cfg.continual.n_task):

        # 現在タスクの更新
        cfg.continual.target_task = target_task
        print('Start Training current task {}'.format(cfg.continual.target_task))


        # model2 のパラメーターを model1 のパラメータで上書きして固定
        model2 = copy.deepcopy(model)

        # =====================================================
        # リプレイバッファ内にあるデータのインデックスを獲得
        # rank0のみで処理を実行し，各rankへ replay_indicesを配布する
        # =====================================================
        replay_indices = set_buffer(cfg, model, prev_indices=replay_indices)


        # バッファ内データのインデックスを保存（検証や分析時に読み込むため）
        if cfg.ddp.local_rank == 0:
            save_replay_indices_to_txt(replay_indices=replay_indices,
                                       save_path=os.path.join(cfg.log.mem_path, 'task_{target_task:03d}_replay.txt'.format(target_task=target_task))
            )


        # =====================================================
        # データローダの作成
        # =====================================================
        dataloader = set_loader(cfg, model, replay_indices)


        # =====================================================
        # タスク開始後の前処理（ERなどは必要ないので後回し）
        # =====================================================
        a = 1


        # =====================================================
        # 学習率schedulerの作成
        # =====================================================
        # 訓練前にエポック数を設定（初期エポックだけエポック数を変える場合に必要）
        if target_task == 0 and cfg.optimizer.train.start_epoch is not None:
            cfg.optimizer.train.epochs = cfg.optimizer.train.start_epoch
        else:
            cfg.optimizer.train.epochs = original_epochs
        scheduler = make_scheduler(cfg, cfg.optimizer.train.epochs, dataloader, optimizer)


        # # =====================================================
        # # 学習の実行
        # # =====================================================
        # for epoch in range(1, cfg.optimizer.train.epochs+1):

        #     train(cfg=cfg, model=model, model2=model2, criterion=criterion, optimizer=optimizer, scheduler=scheduler, dataloader=dataloader, epoch=epoch)

        #     # modelのパラメータを保存
        #     if cfg.ddp.local_rank == 0:

        #         # 分析・評価表の保存
        #         dir_path = f"{cfg.log.model_path}/task{cfg.continual.target_task:02d}"
        #         file_path = f"{dir_path}/model_epoch{epoch:03d}.pth"
        #         torch.save(model.module.state_dict(), "model_weights.pth")

        #         # 学習途中から再開するための保存
        #         # （後から実装予定）
            
        
        # # =====================================================
        # # タスク終了時の後処理（ERなどは必要ないので後回し）
        # # =====================================================
        # b = 1


        # # =====================================================
        # # タスク終了時のモデルパラメータを保存
        # # =====================================================
        # file_path = f"{cfg.log.model_path}/model_{cfg.continual.target_task:02d}.pth"
        # save_model(model, optimizer, cfg, cfg.epochs, file_path)


if __name__ == '__main__':
    main()


