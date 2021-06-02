# Online Label Smoothing

[![Build Status](https://www.travis-ci.com/ankandrew/online-label-smoothing-pt.svg?branch=main)](https://www.travis-ci.com/ankandrew/online-label-smoothing-pt)

Pytorch implementation of Online Label Smoothing (OLS) presented in [_**Delving Deep into Label Smoothing**_](https://arxiv.org/abs/2011.12562).

## Introduction

As the abstract states, **OLS** is a strategy to generates **soft labels** based on
the statistics of the model prediction for the target category. The core idea is that
instead of using fixed **soft labels** for every epoch, we go updating them based on
the stats of **correct** predicted samples.

More details and experiment results can be found in the [paper](https://arxiv.org/abs/2011.12562).

## Usage

Usage of [**OnlineLabelSmoothing**](./ols/online_label_smooth.py) is pretty straightforward.
Just use it as you would use PyTorch [**CrossEntropyLoss**](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
The only thing that is different is that at the end of the epoch you should call `OnlineLabelSmoothing.next_epoch()` so that it internally updates the
`OnlineLabelSmoothing.supervise` matrix that will be used in the next epoch for the _**soft labels**_.


### Standalone
```python
from ols import OnlineLabelSmoothing
import torch

k = 4  # Number of classes
b = 32  # Batch size
criterion = OnlineLabelSmoothing(alpha=0.5, n_classes=k, smoothing=0.1)
logits = torch.randn(b, k)  # Predictions
y = torch.randint(k, (b,))  # Ground truth

loss = criterion(logits, y)
```

### PyTorch

```python
from ols import OnlineLabelSmoothing

criterion = OnlineLabelSmoothing(alpha=..., n_classes=...)
for epoch in range(...):  # loop over the dataset multiple times
    for i, data in enumerate(...):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} finished!')
    # Update the soft labels for next epoch
    criterion.next_epoch()
```

### PyTorchLightning

With PL you can simple call `next_epoch()` at the end of the epoch with:

```python
import pytorch_lightning as pl
from ols import OnlineLabelSmoothing

class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = OnlineLabelSmoothing(alpha=..., n_classes=...)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, train_batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        self.criterion.next_epoch()

```

## Installation

```
pip install -r requirements.txt
```

## Citation

```
@article{bochkovskiy2020yolov4,
  title={{YOLOv4}: Optimal Speed and Accuracy of Object Detection},
  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2004.10934},
  year={2020}
}
```
