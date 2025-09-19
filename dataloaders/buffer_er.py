import random
import math
import numpy as np
from collections import defaultdict

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

import torch.distributed as dist

from dataloaders.dataset_er import ImageNet21K_ER
from dataloaders.replay_utils import bcast_from_main_pyobj



def allocate_capacity_safely(remaining_capacity, n_classes):
    """
    残り容量をクラスに安全に割り当てる。
    合計が必ずremaining_capacityを超えないようにする。
    """
    # 各クラスへの最低割り当て
    base_cap = remaining_capacity // n_classes

    # 余りを計算
    remainder = remaining_capacity - (base_cap * n_classes)

    # 余り分だけ+1をするクラスをランダムに決定
    extra_indices = set(random.sample(range(n_classes), remainder))

    # 各クラスへの最終割り当てを決定
    alloc_list = []
    for idx in range(n_classes):
        if idx in extra_indices:
            alloc_list.append(base_cap + 1)
        else:
            alloc_list.append(base_cap)

    # 合計が必ずremaining_capacityと一致
    assert sum(alloc_list) == remaining_capacity, "割り当てがバッファサイズと一致していません！"
    return alloc_list


# ring buffer
def set_replay_samples_ring(cfg, model, prev_indices=None):

    is_training = model.training
    model.eval()


    # =================================================================
    # 現状のリプレイバッファの内容を削除して新しいデータを格納する領域を確保する
    # =================================================================
    if prev_indices is None:
        
        prev_indices = []
        # observed_classes = list(range(0, cfg.continual.cls_per_task[0]))
        observed_classes = []
    
    else:

        # ==============================================================
        # データ選択のために仮データセットを作成
        # ==============================================================
        # データ拡張の定義
        transform = transforms.Compose([
            transforms.Resize(cfg.dataset.size),
            transforms.ToTensor(),
        ])

        # データセット作成処理
        if cfg.dataset.type == 'imagenet21k':
            
            subset_indices = []
            
            # データ選択の対象となる一つ前のタスク
            target_task = cfg.continual.target_task - 1
            print("target_task: ", target_task)
            
            # データセットの作成
            train_dataset = ImageNet21K_ER(cfg, transforms=transform, target_task=target_task, train=True, replay=False)
            
            # データ選択の対象となる新しいタスクの「ファイルパス＋ラベル」と「ラベル」
            target_files = train_dataset.all_current_files    # ファイルパスとラベルを要素として格納したリスト
            target_labels = train_dataset.all_labels          # ラベルのみを要素として格納したリスト

        else:
            raise ValueError('dataset not supported: {}'.format(cfg.dataset.type))



        # 既にバッファ内にデータが存在する場合，新しいタスクのデータを保存するためにデータを削除
        if len(prev_indices) > 0:

            # ==============================================================
            # 過去タスクのデータに割り当てるバッファのサイズを計算
            # ==============================================================
            shrink_size = None
            new_task_ncls = len(list(range(sum(cfg.continual.cls_per_task[target_task:target_task+1]))))
            old_task_ncls = len(list(range(sum(cfg.continual.cls_per_task[:target_task]))))
            total_classes = old_task_ncls + new_task_ncls

            # 新しく保存するクラスの数と保存してあるクラスの数
            print("new_task_ncls: ", new_task_ncls)
            print("old_task_ncls: ", old_task_ncls)
            print("total_classes: ", total_classes)

            # 新旧タスクのバッファ割合を決定
            old_capacity = int(cfg.continual.mem_size * (old_task_ncls / total_classes))
            new_capacity = cfg.continual.mem_size - old_capacity  # 残りを新データに割り当て
            print(f"[INFO] Old data capacity: {old_capacity}, New data capacity: {new_capacity}")

            
            # ========== 既存バッファを縮小 ==========
            class_to_prev = defaultdict(list)
            for file_str in prev_indices:
                cls_id = int(file_str.split(" ")[-1])  # ラベルは末尾
                class_to_prev[cls_id].append(file_str)

            base_per_class = old_capacity // old_task_ncls
            remainder = old_capacity - (base_per_class * old_task_ncls)

            new_prev_indices = []

            # for cls_id, files in class_to_prev.items():
            #     if len(files) <= base_per_class:
            #         selected = files
            #     else:
            #         selected = random.sample(files, base_per_class)
            #     new_prev_indices.extend(selected)


            for cls_id, files in class_to_prev.items():
                if len(files) <= base_per_class:
                    # データ数が少なければ全て選択
                    selected = files
                    class_to_prev[cls_id] = []  # 全部選んだので空にする
                else:
                    # base_per_class 枚だけランダムに選択
                    selected = random.sample(files, base_per_class)
                    # class_to_prev から選択済みを削除
                    class_to_prev[cls_id] = list(set(files) - set(selected))

                new_prev_indices.extend(selected)

            
            print("len(new_prev_indices): ", len(new_prev_indices))   # len(new_prev_indices):  10895

            # 余り分をランダムに追加
            while len(new_prev_indices) < old_capacity:
                for cls_id, files in class_to_prev.items():
                    if not files:  # そのクラスに追加可能なデータがもう無い
                        continue
                    # 1枚追加
                    new_prev_indices.append(files.pop(random.randrange(len(files))))
                    if len(new_prev_indices) == old_capacity:
                        break


            print("len(new_prev_indices): ", len(new_prev_indices))   # len(new_prev_indices):  10934
            print("old_capacity: ", old_capacity)                     # old_capacity:  11084
            assert len(new_prev_indices) == old_capacity, "縮小結果が想定と一致しません！"

        

        # ==============================================================
        # 現状学習済みのクラスのリストを作成
        # ==============================================================
        observed_classes = list(range(sum(cfg.continual.cls_per_task[:target_task+1])))
    
    
    # 学習済みクラスの確認
    # print("observed_classes: ", observed_classes)


    # =================================================================
    # 学習済みのクラス（前回タスク）がない場合終了
    # =================================================================
    # （taregt_task == 0 の場合に終了）
    if len(observed_classes) == 0:
        return prev_indices


    # =================================================================
    # 確保した領域に新しいデータを格納する
    # クラス毎に学習用データ数が異なるため，一律で保存データ数を決定するのは不可能
    # =================================================================
    # 各クラスのファイルパスを格納したリスト保存する辞書を初期化
    class_to_files = defaultdict(list)

    # ファイルパスを
    for file_str, label in zip(target_files, target_labels):
        # file_path = file_str.split(" ")[0]  # ファイルパス部分だけ抽出
        class_to_files[label].append(file_str)

    # 新しいクラスを保存するためのバッファサイズ
    # remaining_capacity = cfg.continual.mem_size
    if cfg.continual.target_task == 1:
        remaining_capacity = cfg.continual.mem_size
        new_prev_indices = []
    elif cfg.continual.target_task > 1:
        remaining_capacity = new_capacity

    # バッファ結果を格納する辞書
    new_buffer = defaultdict(list)

    # 残りクラス（まだ全データが保存されていないクラス）
    remaining_classes = dict(class_to_files)
    # print("remaining_classes.keys(): ", remaining_classes.keys())


    # ===== 繰り返し：少数クラスを優先的に確定していく =====
    while remaining_classes and remaining_capacity > 0:
        n_classes = len(remaining_classes)
        if n_classes == 0:
            break

        # 各クラスに割り当てる基準容量（浮動小数点）
        per_class_cap_float = remaining_capacity / n_classes
        base_cap = math.floor(per_class_cap_float)
        frac = per_class_cap_float - base_cap  # 小数部分

        next_remaining_classes = {}

        
        # 各クラスへの安全な割り当て
        alloc_list = allocate_capacity_safely(remaining_capacity, len(remaining_classes))

        # クラスと割り当て容量を対応付けて処理
        for (cls_id, files), alloc_size in zip(remaining_classes.items(), alloc_list):
            if len(files) <= alloc_size:
                # 全部保存
                new_buffer[cls_id].extend(files)
                remaining_capacity -= len(files)
            else:
                # ランダムにalloc_size分だけ保存
                chosen = random.sample(files, alloc_size)
                new_buffer[cls_id].extend(chosen)
                remaining_capacity -= alloc_size

                # 保存されなかった分を次ループに回す
                remaining_files = list(set(files) - set(chosen))
                if remaining_files:
                    next_remaining_classes[cls_id] = remaining_files

                
        
        # 残りクラスを更新
        remaining_classes = next_remaining_classes
    

    # =================================================================
    # 確保した領域に新しいデータを格納する
    # =================================================================
    # バッファ結果を確認（デバッグ用）
    total_saved = sum(len(v) for v in new_buffer.values())
    print(f"Total saved samples in buffer: {total_saved} / {cfg.continual.mem_size}")
    for cls_id, files in new_buffer.items():
        print(f" Class {cls_id}: {len(files)} samples")

    # ファイルパスをリストとして返却（各クラスを結合）
    final_buffer_list = []
    for cls_id, files in new_buffer.items():
        for f in files:
            # final_buffer_list.append(f + " " + str(cls_id))
            final_buffer_list.append(f)


    model.training = is_training


    # ============================== 確認用 ==============================
    # print("len(final_buffer_list): ", len(final_buffer_list))
    # print("len(set(final_buffer_list)): ", len(set(final_buffer_list)))
    # # print("final_buffer_list[0]: ", final_buffer_list[0])

    # for file in final_buffer_list:
    #     print("file: ", file)

    # list_a = ["a", "a", "c", "d",]
    # print("len(list_a): ", len(list_a))
    # print("len(set(list_a)): ", len(set(list_a)))
    # ============================== 確認用 ==============================


    return new_prev_indices + final_buffer_list


