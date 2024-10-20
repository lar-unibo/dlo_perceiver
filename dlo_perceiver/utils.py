import torch
import torch.nn.functional as F


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
            for base_lr in self.base_lrs
        ]


def warmup_learning_rate(optimizer, global_step, config):
    if config["warmup_steps"] > 0 and global_step <= config["warmup_steps"]:
        lr_update = (config["lr"] - config["warmup_lr"]) * float(global_step / config["warmup_steps"])
        lr = config["warmup_lr"] + lr_update
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class WeightedFocalLoss(torch.nn.Module):

    def __init__(self, alpha=0.25, gamma=2, device="cuda"):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none").view(-1)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)

        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
