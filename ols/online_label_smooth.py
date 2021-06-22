import torch
import torch.nn as nn
from torch import Tensor


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
        assert 0 <= alpha <= 1, 'Alpha must be in range [0, 1]'
        self.a = alpha
        self.n_classes = n_classes
        # Initialize soft labels with normal LS for first epoch
        self.register_buffer('supervise', torch.zeros(n_classes, n_classes))
        self.supervise.fill_(smoothing / (n_classes - 1))
        self.supervise.fill_diagonal_(1 - smoothing)

        # Update matrix is used to supervise next epoch
        self.register_buffer('update', torch.zeros_like(self.supervise))
        # For normalizing we need a count for each class
        self.register_buffer('idx_count', torch.zeros(n_classes))
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, y_h: Tensor, y: Tensor):
        # Calculate the final loss
        soft_loss = self.soft_loss(y_h, y)
        hard_loss = self.hard_loss(y_h, y)
        return self.a * hard_loss + (1 - self.a) * soft_loss

    def soft_loss(self, y_h: Tensor, y: Tensor):
        """
        Calculates the soft loss and calls step
        to update `update`.

        :param y_h: Predicted logits.
        :param y: Ground truth labels.

        :return: Calculates the soft loss based on current supervise matrix.
        """
        y_h = y_h.log_softmax(dim=-1)
        if self.training:
            with torch.no_grad():
                self.step(y_h.exp(), y)
        true_dist = torch.index_select(self.supervise, 1, y).swapaxes(-1, -2)
        return torch.mean(torch.sum(-true_dist * y_h, dim=-1))

    def step(self, y_h: Tensor, y: Tensor) -> None:
        """
        Updates `update` with the probabilities
        of the correct predictions and updates `idx_count` counter for
        later normalization.

        Steps:
            1. Calculate correct classified examples.
            2. Filter `y_h` based on the correct classified.
            3. Add `y_h_f` rows to the `j` (based on y_h_idx) column of `memory`.
            4. Keep count of # samples added for each `y_h_idx` column.
            5. Average memory by dividing column-wise by result of step (4).

        Note on (5): This is done outside this function since we only need to
                     normalize at the end of the epoch.
        """
        # 1. Calculate predicted classes
        y_h_idx = y_h.argmax(dim=-1)
        # 2. Filter only correct
        mask = torch.eq(y_h_idx, y)
        y_h_c = y_h[mask]
        y_h_idx_c = y_h_idx[mask]
        # 3. Add y_h probabilities rows as columns to `memory`
        self.update.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        # 4. Update `idx_count`
        self.idx_count.index_add_(0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32))

    def next_epoch(self) -> None:
        """
        This function should be called at the end of the epoch.

        It basically sets the `supervise` matrix to be the `update`
        and re-initializes to zero this last matrix and `idx_count`.
        """
        # 5. Divide memory by `idx_count` to obtain average (column-wise)
        self.idx_count[torch.eq(self.idx_count, 0)] = 1  # Avoid 0 denominator
        # Normalize by taking the average
        self.update /= self.idx_count
        self.idx_count.zero_()
        self.supervise = self.update
        self.update = self.update.clone().zero_()
