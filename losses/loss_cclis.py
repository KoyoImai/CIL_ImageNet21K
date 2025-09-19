from __future__ import print_function

import torch
import torch.nn as nn


class ISSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, prototypes_mode='mean',
                 base_temperature=0.07, embedding_shape=512, opt=None):
        super(ISSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.prototypes_mode = prototypes_mode
        
        self.n_cls = opt.continual.n_cls 
        self.mem_size = opt.continual.mem_size
        self.cls_per_task = opt.continual.cls_per_task
        
        self.embedding_shape = embedding_shape

        self.wo_is = opt.method.hypara.wo_is


    def score_calculate(self, output, features, labels=None, importance_weight=None, index=None, target_labels=None, sample_num=[], mask=None, score_mask=[], all_labels=None):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        # features: [batch_size, embed_size]
        self.replay_sample_num = torch.tensor(sample_num)
        cur_all_labels = torch.arange(target_labels[-1] + 1)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu')) 
        
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)

        # バッチサイズ
        batch_size = features.shape[0]
        
        # 
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            raise ValueError('`labels` or `mask` should be defined')
        elif labels is not None:
            labels = labels.contiguous()  # labels [batch_size]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            all_labels = torch.unique(labels).view(-1, 1).to(device)  
            cur_all_labels = cur_all_labels.view(-1, 1).to(device)  
            mask = torch.eq(all_labels, labels.T).float().to(device) 
        else:
            mask = mask.float().to(device)
        
        importance_weight = importance_weight.float().to(device)
        
        if all_labels is not None:
            output = output[:target_labels[-1] + 1, :]
                
        # compute logits
        logits = torch.div(output, self.temperature).to(torch.float64)  # [class_num, batch_size]
        with torch.no_grad():
            cur_class_num = target_labels[-1] + 1
            batch_score_sum = torch.zeros(cur_class_num, cur_class_num)
            score_mat = torch.exp(logits)

            for idx in range(target_labels[-1] + 1):
                batch_score_sum[:, idx] = score_mat[:, labels==idx].sum(1)

        return score_mat, batch_score_sum


    def forward(self, output, features, labels=None, importance_weight=None, index=None, target_labels=None, sample_num=None, mask=None, score_mask=None, all_labels = None, reduction='mean'):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        """
            sample_num   : 各クラスのデータの数が格納されているlist
            target_labels: 現在タスクのラベル
        """
        
        
        #  形状確認
        # print("output.shape: ", output.shape)        # output.shape:  torch.Size([10, 512])
        # print("features.shape: ", features.shape)    # features.shape:  torch.Size([512, 128])
        # print("labels.shape: ", labels.shape)        # labels.shape:  torch.Size([512])
        # print("importance_weight.shape: ", importance_weight.shape)   # importance_weight.shape:  torch.Size([512])
        # print("sample_num: ", sample_num)        # sample_num:  [ 250  250 5000 5000]など
        # print("target_labels: ", target_labels)  # trget_labels:  [0, 1]など
        # print("score_mask.shape: ", score_mask.shape)
        # if score_mask is not None:
        #     print("scpre_mask.shape: ", score_mask.shape)
        #     assert False

    

        self.replay_sample_num = torch.tensor(sample_num) if sample_num is not None else torch.Tensor()
        _cur_all_labels = torch.arange(target_labels[-1] + 1)

        # assert False

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu')) 


        # featuresの形状を確認して想定外の形状ならエラー
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)


        # バッチサイズ
        batch_size = features.shape[0]
        
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            raise ValueError('`labels` or `mask` should be defined')
        
        # ラベルが与えられる場合
        elif labels is not None:

            # メモリ配置を連続に（後のエラー対策）
            labels = labels.contiguous()  # labels [batch_size]
            
            # 形状確認で不適切ならエラー発生
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            
            # 重複ラベルを削除して，1次元ベクトルに変換
            all_labels = torch.unique(labels).view(-1, 1).to(device)
            # print("all_labels.shape: ", all_labels.shape)  # all_labels.shape:  torch.Size([2, 1])
            
            # マスクの作成
            mask = torch.eq(all_labels, labels.T).float().to(device)
            # print("mask.shape: ", mask.shape)   # mask.shape:  torch.Size([2, 512])
            
        else:
            mask = mask.float().to(device)


        # 重要度重み
        importance_weight = importance_weight.float().to(device)
        # print("importance_weight.shape: ", importance_weight.shape)   # importance_weight.shape:  torch.Size([512])

        # print("self.n_cls: ", self.n_cls)   # self.n_cls:  10
        # print("torch.zeros(len(all_labels), self.n_cls).float().to(device).shape: ",
        #       torch.zeros(len(all_labels), self.n_cls).float().to(device).shape)       # torch.zeros(len(all_labels), self.n_cls).float().to(device).shape:  torch.Size([2, 10])


        # ラベルが与えられていた場合実行
        if all_labels is not None:

            # バッチに含まれるクラスに対応した出力だけ取り出すマスクの作成
            prototypes_mask = torch.scatter(
                torch.zeros(len(all_labels), self.n_cls).float().to(device),   # 形状[現在のラベルの種類，学習中のデータセットにおけるクラス数]のゼロ行列
                1,
                all_labels.view(-1, 1).to(device),
                1
            )

            # print("output.shape: ", output.shape)               # )
            # print("prototypes_mask.shape: ", prototypes_mask.shape)   # prototypes_mask.shape:  torch.Size([2, 10])
            # print("prototypes_mask: ", prototypes_mask)

            # バッチに含まれるクラスの出力のみを取り出す
            output = torch.matmul(prototypes_mask, output)
            # print("output.shape: ", output.shape)   # output.shape:  torch.Size([2, 512])

        # logitsを計算（s_{i,j} = c_{i} * z_{j} / \tau）
        anchor_dot_contrast = torch.div(output, self.temperature).to(torch.float64) # [class_num, batch_size]
        # print("anchor_dot_contrast.shape: ", anchor_dot_contrast.shape)  # anchor_dot_contrast.shape:  torch.Size([2, 512])

        # 数値的安定性のため，各logitsから最大値を引く
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 各アンカーから最大の値を取り出す
        logits = anchor_dot_contrast - logits_max.detach()                   # 最大値を引く

        # print("logits_max.shape: ", logits_max.shape)   # logits_max.shape:  torch.Size([2, 1])
        

        # どのタスクに属するかのラベル（task id）を取得
        task_all_labels = all_labels // self.cls_per_task
        task_labels = labels // self.cls_per_task
        # print("task_all_labels.shape: ", task_all_labels.shape)   # task_all_labels.shape:  torch.Size([2, 1])
        # print("task_labels.shape: ", task_labels.shape)           # task_labels.shape:  torch.Size([512])

        
        # スコアマスクが与えられた時
        if score_mask is not None:
            # print('score_mask', score_mask)
            label_mask = torch.tensor([score_mask[item] for item in labels.tolist()]).to(device)
            score_scale_mask = torch.eq(all_labels, label_mask).float().to(device)  
        
        # スコアマスクが与えられない時
        else:
            # 形状[現在のラベルの種類，バッチサイズ]の1行列を作成
            score_scale_mask = torch.ones(len(all_labels), len(labels)).to(device)
            # print("score_scale_mask: ", score_scale_mask)
            # print("score_scale_mask.shape: ", score_scale_mask.shape)   # score_scale_mask.shape:  torch.Size([2, 512])


        # 重要度重みの値を補正する
        with torch.no_grad():
            _importance_weight = importance_weight * (mask * mask.sum(dim=1, keepdim=True)).sum(dim=0)
        # print("importance_weight.shape: ", importance_weight.shape)   # importance_weight.shape:  torch.Size([512])
        # print("importance_weight[0:3]: ", importance_weight[0:3])
        # print("_importance_weight[0:3]: ", _importance_weight[0:3])
        # assert False


        # 現在タスクのデータを見極めるためのマスク
        cur_task_mask_col = (task_all_labels != (target_labels[-1] // 2)).float()    # クラスが現タスクかどうか
        cur_task_mask_row = (task_labels != (target_labels[-1] // 2)).float()        # サンプルが現タスクかどうか
        # print("cur_task_mask_col.shape: ", cur_task_mask_col.shape)   # cur_task_mask_col.shape:  torch.Size([2, 1])
        # print("cur_task_mask_row.shape: ", cur_task_mask_row.shape)   # cur_task_mask_row.shape:  torch.Size([512])


        # 重要度重みをかける負例を選択するためのマスクを作成
        cur_task_mask = (cur_task_mask_col.view(-1,1) * cur_task_mask_row).to(device)
        # torch.ones_like(mask) - mask：負例部分を1とするマスク
        all_mask = score_scale_mask * cur_task_mask * (torch.ones_like(mask) - mask)

        # print("cur_task_mask.shape: ", cur_task_mask.shape)   # cur_task_mask.shape:  torch.Size([2, 512])


        
        ## self.wo_isがTrueなら重要度スコアによる補正をなくし，単純なプロトタイプベースの対照学習を実行
        if not self.wo_is:
            # (s_{i,j} - log())
            _logits = logits - torch.log(_importance_weight) * all_mask
            # print("_logits.shape: ", _logits.shape)    # _logits.shape:  torch.Size([2, 512])
            log_prob = logits - torch.log(torch.exp(_logits).sum(1, keepdim=True))  # normalize
            # print("log_prob.shape: ", log_prob.shape)  # log_prob.shape:  torch.Size([2, 512])
        else:
            log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))

        if reduction == "mean":
            IS_supcon_loss = - (log_prob * mask).sum() / mask.sum()
        elif reduction == "grad_analysis":
            IS_supcon_loss = - (log_prob * mask).sum(0) / mask.sum()
            # print("IS_supcon_loss.shape: ", IS_supcon_loss.shape)  # IS_supcon_loss.shape:  torch.Size([512])

        return IS_supcon_loss