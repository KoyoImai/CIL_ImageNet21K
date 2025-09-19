



from schedulers.multisteplr import MultiStepWarmupScheduler
from schedulers.cosineannealing import CosineAnnealingWarmupScheduler




def make_scheduler(cfg, epochs, dataloader, optimizer):

    if cfg.method.name == "er" and cfg.scheduler.type == "multisteplr":

        # warmupステップ数の決定
        steps_per_epoch = len(dataloader)

        if cfg.continual.target_task != 0:
            scheduler = MultiStepWarmupScheduler(optimizer=optimizer,
                                                 warmup_epochs=cfg.scheduler.warmup_epochs,
                                                 milestones=cfg.scheduler.milestones,
                                                 steps_per_epoch=steps_per_epoch,
                                                 gamma=0.1, last_epoch=-1
                                                 )
        else:
            scheduler = MultiStepWarmupScheduler(optimizer=optimizer,
                                                 warmup_epochs=0,
                                                 milestones=cfg.scheduler.milestones,
                                                 steps_per_epoch=steps_per_epoch,
                                                 gamma=0.1, last_epoch=-1
                                                 )
    elif cfg.method.name == "er" and cfg.scheduler.type == "cosine":

        # warmupステップ数の決定
        steps_per_epoch = len(dataloader)

        # schedulerの設定
        if cfg.continual.target_task != 0:
            scheduler = CosineAnnealingWarmupScheduler(optimizer,
                                                       warmup_epochs=cfg.scheduler.warmup_epochs,
                                                       total_epochs=epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       min_lr=1e-5)

        else:
            scheduler = CosineAnnealingWarmupScheduler(optimizer,
                                                       warmup_epochs=0,
                                                       total_epochs=epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       min_lr=1e-5)

    else:
        assert False



    return scheduler




