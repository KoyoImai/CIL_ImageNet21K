


import torch



def pre_process(cfg, model, model2, optimizer, dataloader):

    if cfg.method.name in ["er"]:

        # input = torch.randn([10, 3, 224, 224]).cuda()
        # pre_output = model(input)
        # print("pre_output.shape: ", pre_output.shape)
        # print("pre_output[0][0:10]: ", pre_output[0][0:10])


        if cfg.continual.target_task == 0:
            return 
        else:
            # fc層を追加
            new_params = model.update_fc(cfg=cfg)
            if torch.cuda.is_available():
                model = model.cuda()

            # print("optimizer.param_groups: ", optimizer.param_groups)
            # print("optimizer.param_groups.keys(): ", optimizer.param_groups.keys())

            # # optimizer の momentum を確認
            # ref_p = next(iter(optimizer.state))  # 既存パラメータの1つ
            # print("ref_p: ", ref_p)
            # m_before = optimizer.state[ref_p]['momentum_buffer'].clone()
            # print("m_before: ", m_before)
            
            # 追加したfc層をoptimizerの最適化対象に加える
            optimizer.add_param_group({
                "params": new_params,
                "lr": cfg.optimizer.train.learning_rate,
                "momentum": cfg.optimizer.train.momentum,
                "weight_decay": cfg.optimizer.train.weight_decay,
            })
            optimizer.zero_grad(set_to_none=True)



        # post_output = model(input)
        # print("post_output.shape: ", post_output.shape)
        # print("post_output[0][0:10]: ", post_output[0][0:10])

        # print("model: ", model)
        return














