

import os
import numpy as np


import torch


from utils import write_csv, AverageMeter, write_csv_dict




# 列の順序（ヘッダー）
csv_headers = [
    "epoch",
    "iter",
    "num_iters",
    "batch_size",
    "loss",
    "loss_avg",
    "acc",
    "acc_avg",
    "lr",
]



def train_er(cfg, model, model2, criterion, optimizer, scheduler, dataloader, epoch):

    # modelをtrainモード，model2をevalモードに変更
    model.train()
    model2.eval()

    # 学習記録
    losses = AverageMeter()
    accuracies = AverageMeter()

    # print("cfg.continual.cls_per_task: ", cfg.continual.cls_per_task)
    # assert False
    

    corr = [0.] * sum(cfg.continual.cls_per_task[:cfg.continual.target_task])
    cnt  = [0.] * sum(cfg.continual.cls_per_task[:cfg.continual.target_task])
    correct_task = 0.0


    for idx, (images, labels, _, meta) in enumerate(dataloader):
        
        if cfg.ddp.local_rank == 0:
            if idx == 0:
                print("meta['files'][0]: ", meta['files'][0])

        # gpuが使用可能ならgpu上に配置
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        # print("images.shape: ", images.shape)
        # print("labels[0]: ", labels[0])
        # print("meta['files'][0]: ", meta['files'][0])
        # print()


        # バッチサイズの取得
        bsz = labels.shape[0]

        # モデルにデータを入力して出力を取得
        y_pred = model(images)
        # print("y_pred.shape: ", y_pred.shape)

        # 損失計算
        loss = criterion(y_pred, labels).mean()

        # update metric
        losses.update(loss.item(), bsz)


        # 正解率の計算
        preds = y_pred.argmax(dim=1)               # 予測ラベル
        correct = preds.eq(labels).sum().item()    # 正解数
        
        # accuracy
        acc = correct / bsz
        accuracies.update(acc, bsz)

        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']


        # ===== CSV に1行追記 =====
        if cfg.ddp.local_rank == 0:
            row = {
                "epoch":      int(epoch),
                "iter":       int(idx + 1),
                "num_iters":  int(len(dataloader)),
                "batch_size": int(bsz),
                "loss":       float(losses.val),
                "loss_avg":   float(losses.avg),
                "acc":        float(accuracies.val),
                "acc_avg":    float(accuracies.avg),
                "lr":         float(current_lr),
            }
            write_csv_dict(f"{cfg.log.explog_path}/explog.csv", row, headers=csv_headers)


        # 学習状況の表示
        if (idx % cfg.print_freq == 0) and (cfg.ddp.local_rank == 0):
            print("Train: [{0}][{1}/{2}] "
                  "Loss: {loss.val:.3f} ({loss.avg:.3f}) "
                  "Acc: {acc.val:.3f} ({acc.avg:.3f}) "
                  "LR: {lr:.5f}".format(
                      epoch, idx + 1, len(dataloader),
                      loss=losses,
                      acc=accuracies,
                      lr=current_lr,
                  ))
            
        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    

    return losses












