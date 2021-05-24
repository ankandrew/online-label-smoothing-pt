import unittest

import torch
import torch.nn as nn
from torch.autograd import Variable

from ols.label_smooth import LabelSmoothingLoss


class TestLoss(unittest.TestCase):
    def test_hard_loss(self):
        crit = LabelSmoothingLoss(classes=5, smoothing=0)
        crit2 = nn.CrossEntropyLoss()
        predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                     [0, 0.9, 0.2, 0.2, 1],
                                     [1, 0.2, 0.7, 0.9, 1]])
        loss1 = crit(Variable(predict), Variable(torch.LongTensor([2, 1, 0])))
        loss2 = crit2(Variable(predict), Variable(torch.LongTensor([2, 1, 0])))
        self.assertEqual(loss1, loss2)


if __name__ == '__main__':
    unittest.main()
