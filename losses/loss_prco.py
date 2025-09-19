
from __future__ import print_function

import torch
import torch.nn as nn






class ProtoSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, prototypes_mode='mean',
                 base_temperature=0.07, embedding_shape=512, opt=None):
        
        super(ProtoSupConLoss, self).__init__()

        self.temperature = temperature

        self.n_cls = opt.continual.n_cls 
        self.mem_size = opt.continual.mem_size
        self.cls_per_task = opt.continual.cls_per_task
        
        self.embedding_shape = embedding_shape


    def forward(self, output, features, labels, index, target_labels):


        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        # バッチサイズ
        batch_size = features.shape[0]


        # featuresの形状を確認して想定外の形状ならエラー
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)
        

        # ラベルから mask を作成
        # メモリ配置を連続に（後のエラー対策）
        labels = labels.contiguous()

        # ラベルの数がバッチサイズと一致するかを確認
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # 全てのラベル種類を取得
        all_labels = torch.unique(labels).view(-1, 1).to(device)
        # print("all_labels.shape: ", all_labels.shape)    # all_labels.shape:  torch.Size([2, 1])
        
        # マスクの作成
        mask = torch.eq(all_labels, labels.T).float().to(device)
        # print("mask.shape: ", mask.shape)     # mask.shape:  torch.Size([2, 512])
    
        # バッチに含まれるクラスに対応した output(プロトタイプとfeaturesの内積) 取り出すためマスクの作成
        prototypes_mask = torch.scatter(
            torch.zeros(len(all_labels), self.n_cls).float().to(device),   # 形状[現在のラベルの種類，学習中のデータセットにおけるクラス数]のゼロ行列
            1,
            all_labels.view(-1, 1).to(device),
            1
        )
        # print("prototypes_mask.shape: ", prototypes_mask.shape)    # prototypes_mask.shape:  torch.Size([2, 10])

        # バッチに含まれるクラスの出力のみを取り出す
        # print("output.shape: ", output.shape)    # output.shape:  torch.Size([10, 512])
        output = torch.matmul(prototypes_mask, output)
        # print("output.shape: ", output.shape)    # output.shape:  torch.Size([2, 512])

        # logitsを計算（s_{i,j} = c_{i} * z_{j} / \tau）
        logits = torch.div(output, self.temperature).to(torch.float64) # [class_num, batch_size]
        # print("anchor_dot_contrast.shape: ", anchor_dot_contrast.shape)  # anchor_dot_contrast.shape:  torch.Size([2, 512])

        # 数値的安定性のため，各logitsから最大値を引く
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)  # 各アンカーから最大の値を取り出す
        logits = logits - logits_max.detach()                   # 最大値を引く


        # 通常の log_softmax
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))

        loss = - (log_prob * mask).sum() / mask.sum()

        return loss
    

