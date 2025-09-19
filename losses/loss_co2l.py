import torch
import torch.nn as nn
import torch.nn.functional as F


'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
'''


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, not_asym=False):
        super(SupConLoss, self).__init__()

        self.temperature = temperature    
        self.contrast_mode = contrast_mode 
        self.base_temperature = base_temperature
        self.not_asym = not_asym

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # バッチサイズ
        batch_size = features.shape[0]
        
        # ラベルとマスクの処理
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # print("labels.shape: ", labels.shape)  # labels.shape:  torch.Size([512, 1])
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            
            # 同一ラベルのペアを1に，異なるラベルのペアを0にするマスクを作成
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # crop数の取得
        contrast_count = features.shape[1]

        # featuresの1次元目を削除し，連結する．
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print("contrast_feature.shape: ", contrast_feature.shape)  # contrast_feature.shape:  torch.Size([1024, 128]
        
        # アンカー特徴量とアンカー数の決定（defaultはall）
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # logitsの計算
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print("anchor_dot_contrast.shape: ", anchor_dot_contrast.shape)  # anchor_dot_contrast.shape:  torch.Size([1024, 1024])

        
        # 数値的安定性のために最大値を引く
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # maskの作成
        mask = mask.repeat(anchor_count, contrast_count)
        # print("mask.shape: ", mask.shape)   # mask.shape:  torch.Size([1024, 1024])
        
        # 自身との比較を除外するためのマスク
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # ラベルから自身との比較を除外
        mask = mask * logits_mask

        # 対数確率の分子・分母を計算
        exp_logits = torch.exp(logits) * logits_mask
        # print("exp_logits.shape: ", exp_logits.shape)  # exp_logits.shape:  torch.Size([1024, 1024])

        # 対数確率を計算
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print("log_prob.shape: ", log_prob.shape)  # log_prob.shape:  torch.Size([1024, 1024])

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print("mean_log_prob_pos.shape: ", mean_log_prob_pos.shape)  # mean_log_prob_pos.shape:  torch.Size([1024])

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print("loss.shape: ", loss.shape)  # loss.shape:  torch.Size([1024])

        
        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        
        # 非対称教師あり対照損失のため，過去クラスを除外
        if not self.not_asym:
            loss = curr_class_mask * loss.view(anchor_count, batch_size)
            # print("loss.shape: ", loss.shape) 
        else:
            loss = loss.view(anchor_count, batch_size)

        if reduction == 'mean':
            loss = loss.mean()
            # print("loss.shape: ", loss.shape)
            
        elif reduction == 'none':
            loss = loss.mean(0)
        
        elif reduction == "grad_analysis":
            return loss
        
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss






class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):

        super(ContrastiveLoss, self).__init__()

        self.temp = temperature

    def forward(self, features, bsz):

        labels = torch.cat([torch.arange(bsz) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temp
        return logits, labels