from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, final_lr):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup phase
                return (epoch + 1) / warmup_epochs
            else:
                # Decay phase after warmup
                return (total_epochs - epoch) / (total_epochs - warmup_epochs)

        super().__init__(optimizer, lr_lambda)
