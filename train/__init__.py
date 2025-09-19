


from train.train_er import train_er





def train(cfg, model, model2, criterion, optimizer, scheduler, dataloader, epoch):


    if cfg.method.name in ["er"]:
        losses = train_er(cfg=cfg, model=model, model2=model2, criterion=criterion, optimizer=optimizer, scheduler=scheduler, dataloader=dataloader, epoch=epoch)
    
    else:
        assert False







