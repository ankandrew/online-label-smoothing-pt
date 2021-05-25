import unittest

import torch

from ols.online_label_smooth import OnlineLabelSmoothing


class TestUpdate(unittest.TestCase):
    def setUp(self) -> None:
        self.k = 3  # Num. classes
        self.y = torch.tensor([1, 0, 0, 1], dtype=torch.long)
        self.y_h = torch.tensor([
            [0.3, 0.6, 0.1],  # Correct
            [0.7, 0.2, 0.1],  # Correct
            [0.3, 0.3, 0.4],
            [0.2, 0.7, 0.1]  # Correct
        ])

    def test_ols_step_next_epoch(self):
        ols = OnlineLabelSmoothing(alpha=0.5, n_classes=3, smoothing=0.1)
        ols.step(self.y_h, self.y)
        # After the step, ols.update should have accumulated correct probabilities
        expected_update = torch.tensor([
            [0.7, 0.5, 0.00],
            [0.2, 1.3, 0.00],
            [0.1, 0.2, 0.00]
        ])
        expected_idx_count = torch.tensor([1, 2, 0])
        self.assertTrue(torch.eq(ols.update, expected_update).all().item())
        self.assertTrue(torch.eq(ols.idx_count, expected_idx_count).all().item())
        # Check after next epoch
        # 1. Check for NaN in `ols.supervise`
        # 2. idx_count is all zero
        ols.next_epoch()
        self.assertTrue(~torch.isnan(ols.supervise).any())
        self.assertTrue((ols.idx_count == 0).all())

    def test_update_logic(self):
        memory = torch.zeros(self.k, self.k)
        idx_count = torch.zeros(self.k)
        '''
        1. Calculate correct classified examples
        2. Filter `y_h` based on the correct classified
        3. Add `y_h_f` rows to the `j` (based on y_h_idx) column of `memory`
        4. Keep count of # samples added for each `y_h_idx` column
        5. Average memory by dividing column-wise by result of step (4)
        '''
        # 1. Calculate predicted classes
        y_h_idx = self.y_h.argmax(dim=-1)  # tensor([1, 0, 2, 1])
        # 2. Filter only correct
        mask = torch.eq(y_h_idx, self.y)
        y_h_c = self.y_h[mask]
        y_h_idx_c = y_h_idx[mask]  # tensor([1, 0, 1])
        # 3. Add y_h probabilities rows as columns to `memory`
        memory.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        # 4. Update `idx_count`
        idx_count.index_add_(0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32))
        # 5. Divide memory by `idx_count` to obtain average (column-wise)
        idx_count[torch.eq(idx_count, 0)] = 1  # Avoid 0 denominator
        memory /= idx_count
        ground_truth = torch.tensor([[
            [0.7, 0.25, 0.00],
            [0.2, 0.65, 0.00],
            [0.1, 0.10, 0.00]
        ]])
        self.assertTrue(torch.eq(ground_truth, memory).all().item())


if __name__ == '__main__':
    unittest.main()
