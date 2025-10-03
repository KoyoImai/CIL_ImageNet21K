
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



def pre_process(cfg, model, model2, optimizer, dataloader, use_resume):

    if cfg.method.name in ["er"]:

        # input = torch.randn([10, 3, 224, 224]).cuda()
        # pre_output = model(input)
        # print("pre_output.shape: ", pre_output.shape)
        # print("pre_output[0][0:10]: ", pre_output[0][0:10])


        if cfg.continual.target_task == 0:
            return model, optimizer
        else:
            
            if not use_resume:
                # ====== 追加: ここで一度全 rank を待機 ======
                if getattr(cfg.ddp, "use_ddp", False) and dist.is_available() and dist.is_initialized():
                    dist.barrier()

                # fc層を追加
                new_params = model.module.update_fc(cfg=cfg)
                if torch.cuda.is_available():
                    model = model.cuda()
                # print("new_params[0].shape: ", new_params[0].shape)
                # assert False
                

                # ====== 追加: fc 追加が全 rank 終わるのを待機 ======
                if getattr(cfg.ddp, "use_ddp", False) and dist.is_available() and dist.is_initialized():
                    dist.barrier()

                    # 新ヘッドの重みを rank0 → 全rank へ同期
                    with torch.no_grad():
                        for p in model.module.head.parameters():
                            dist.broadcast(p.data, src=0)

                    # ====== 追加: ブロードキャスト完了を待機 ======
                    dist.barrier()
                    
                    # ====== ★ DDP を再ラップ（新 Parameter にフックを付け直す） ======
                    # unwrap → 再ラップ（パラメータ本体は同一）
                    m = model.module
                    model = DDP(
                        m.to(cfg.ddp.local_rank),
                        device_ids=[cfg.ddp.local_rank],
                        find_unused_parameters=True,  # 未使用検出も一応 ON（安全側）
                    )
                    dist.barrier()  # 全 rank で再ラップ完了を待機
                    if cfg.ddp.local_rank == 0:
                        print("model: ", model)
                        print("model.module.head.weight.shape: ", model.module.head.weight.shape)

                # 追加した fc を optimizer に登録（全 rank 同一順序で）
                optimizer.add_param_group({
                    "params": new_params,
                    "lr": cfg.optimizer.train.learning_rate,
                    "momentum": cfg.optimizer.train.momentum,
                    "weight_decay": cfg.optimizer.train.weight_decay,
                })
                optimizer.zero_grad(set_to_none=True)


                # ====== 追加: param_group 追加が全 rank 終わるのを待機 ======
                if getattr(cfg.ddp, "use_ddp", False) and dist.is_available() and dist.is_initialized():
                    dist.barrier()



            # post_output = model(input)
            # print("post_output.shape: ", post_output.shape)
            # print("post_output[0][0:10]: ", post_output[0][0:10])

            # print("model: ", model)
        return model, optimizer






def pre_process_resume(cfg, model, model2, optimizer, dataloader, resume_meta):


    if cfg.method.name in ["er"]:

        # optimizer の 最適化対象となるパラメータ形状を復元
        saved_task = resume_meta["start_task"]

        # 現在の target_task を退避
        _orig_task = cfg.continual.target_task

        for t in range(0, saved_task+1):

            cfg.continual.target_task = t

            model, optimizer = pre_process(cfg=cfg, model=model, model2=model2, optimizer=optimizer, dataloader=dataloader, use_resume=False)
        

        # 元に戻す（以降の処理が参照しても大丈夫なように）
        cfg.continual.target_task = _orig_task

        return model, optimizer










