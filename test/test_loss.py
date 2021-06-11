import unittest

import torch
import torch.nn as nn

from ols import LabelSmoothingLoss, OnlineLabelSmoothing


class TestOnlineLabelSmoothing(unittest.TestCase):
    def setUp(self) -> None:
        self.smoothing = 0.1
        self.m, self.k = 50, 3
        self.x, self.y = torch.randn(self.m, self.k), torch.randint(self.k, (self.m,))
        self.ols = OnlineLabelSmoothing(alpha=0.5, n_classes=self.k, smoothing=0.1)

    def test_initial(self):
        """
        Test that supervise is a probability distribution column-wise
        and that update/idx_count are of correct shape
        """
        self.assertTrue(torch.allclose(self.ols.supervise.sum(dim=0), torch.tensor(1.)))
        self.assertEqual(self.ols.supervise.shape, (self.k, self.k))
        self.assertEqual(self.ols.update.shape, self.ols.supervise.shape)
        self.assertEqual(self.ols.idx_count.shape, (self.k,))

    def test_hard_first_epoch(self):
        """
        Loss should be equal to pytorch Cross Entropy Loss with alpha=1.0
        """
        ols = OnlineLabelSmoothing(alpha=1.0, n_classes=self.k)
        cce = torch.nn.CrossEntropyLoss()
        self.assertEqual(cce(self.x, self.y), ols(self.x, self.y))

    def test_soft_first_epoch(self):
        """
        Loss should be like normal LS (for first epoch)
        """
        # Caution: `ls` and `ols` must both apply (smoothing/classes) or (smoothing/(classes-1))
        ls = LabelSmoothingLoss(classes=self.k, smoothing=self.smoothing)
        ols = OnlineLabelSmoothing(alpha=0.0, n_classes=self.k, smoothing=self.smoothing)
        ls_loss = ls(self.x, self.y)
        ols_loss = ols(self.x, self.y)
        self.assertEqual(ls_loss, ols_loss)

    def test_balance_loss_first_epoch(self):
        a = 0.6  # balancing-term for hard/soft
        hard_loss_fn = torch.nn.CrossEntropyLoss()
        soft_loss_fn = LabelSmoothingLoss(classes=self.k, smoothing=self.smoothing)
        expected_loss = a * hard_loss_fn(self.x, self.y) + (1 - a) * soft_loss_fn(self.x, self.y)
        ols_fn = OnlineLabelSmoothing(alpha=a, n_classes=self.k, smoothing=self.smoothing)
        self.assertEqual(ols_fn(self.x, self.y), expected_loss)

    def test_forward_pass(self):
        x, y = torch.randn(self.m, 8 * 8), torch.randint(self.k, (self.m,))
        model = nn.Sequential(
            nn.Linear(8 * 8, 16),
            nn.Linear(16, 32),
            nn.Linear(32, self.k)
        )
        ols = OnlineLabelSmoothing(alpha=0.6, n_classes=self.k, smoothing=self.smoothing)
        y_h = model(x)
        loss = ols(y_h, y)
        loss.backward()


if __name__ == '__main__':
    unittest.main()
