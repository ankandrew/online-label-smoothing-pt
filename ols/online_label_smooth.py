import torch
import torch.nn as nn

from label_smooth import LabelSmoothingLoss


class OnlineLabelSmoothing(nn.Module):
    """
    Implements Online Label Smoothing from paper
    https://arxiv.org/pdf/2011.12562.pdf
    """

    def __init__(self, alpha: float, n_classes: int, smoothing: float = 0.1):
        """
        :param alpha: Term for balancing soft_loss and hard_loss
        :param n_classes: Number of classes of the classification problem
        :param smoothing: Smoothing factor to be used during first epoch in soft_loss
        """
        super(OnlineLabelSmoothing, self).__init__()
        assert 0 <= self.a <= 1, 'Alpha must be in range [0, 1]'
        self.a = alpha
        # Initialize soft labels to normal Label Smoothing for first epoch
        self.soft_supervise = (1 - smoothing) * torch.eye(n_classes) + smoothing / n_classes
        # # With alpha / (n_classes - 1) ----> Alternative
        # self.supervise = torch.zeros(n_classes, n_classes)
        # self.supervise.fill_(smoothing / (n_classes - 1))
        # self.supervise.fill_diagonal_(1 - smoothing)
        self.update = torch.zeros_like(self.supervise)
        self.soft_loss = LabelSmoothingLoss(n_classes, smoothing)
        # self.hard_loss = LabelSmoothingLoss(n_classes, 0)
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, y_hat, y):
        # Calculate the final loss
        soft_loss = self.soft_loss(y_hat, y)
        hard_loss = self.hard_loss(y_hat, y)
        # Update with correct predictions
        self.step(y_hat, y)
        return self.a * hard_loss + (1 - self.a) * soft_loss

    def soft_loss(self, y_hat, y):
        # y_hat = y_hat.softmax(dim=-1)
        # return torch.mean(torch.sum(-true_dist * y_hat, dim=self.dim))
        pass

    def step(self):
        pass

    def __next_epoch(self):
        self.supervise = self.update
        self.update = torch.zeros_like(self.supervise)
