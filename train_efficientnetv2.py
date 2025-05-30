from typing import Type, Any

import torch 
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import instantiate_class
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torchmetrics import MetricCollection, Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch import nn
from torch.utils.data import random_split, DataLoader


config = {
    'seed': 2021, 
    'trainer': {
        'max_epochs': 100,
        'accelerator': "auto",
        'accumulate_grad_batches': 1,
        'fast_dev_run': False,
        'num_sanity_val_steps': 0,
    },
    'data': {
        'dataset_name': 'cifar10',
        'batch_size': 32,
        'num_workers': 4,
        'size': [224, 224],
        'data_root': 'data',
        'valid_ratio': 0.1
    },
    'model':{
        'backbone_init': {
            'model': 'efficientnet_v2_s_in21k',
            'nclass': 0, # do not change this
            'pretrained': True,
            },
        'optimizer_init':{
            'class_path': 'torch.optim.SGD',
            'init_args': {
                'lr': 0.01,
                'momentum': 0.95,
                'weight_decay': 0.0005
                }
            },
        'lr_scheduler_init':{
            'class_path': 'torch.optim.lr_scheduler.CosineAnnealingLR',
            'init_args':{
                'T_max': 0 # no need to change this
                }
            }
    }
}

class BaseDataModule(LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 dataset: Type[Any],
                 train_transform: Type[Any],
                 test_transform: Type[Any],
                 batch_size: int = 64,
                 num_workers: int = 4,
                 data_root: str = 'data',
                 valid_ratio: float = 0.1):
        """
        Base Data Module
        :arg
            Dataset: Enter Dataset
            batch_size: Enter batch size
            num_workers: Enter number of workers
            size: Enter resized image
            data_root: Enter root data folder name
            valid_ratio: Enter valid dataset ratio
        """
        super(BaseDataModule, self).__init__()
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.valid_ratio = valid_ratio
        self.num_classes = None
        self.num_step = None
        self.prepare_data()

    def prepare_data(self) -> None:
        train = self.dataset(root=self.data_root, train=True, download=True)
        test = self.dataset(root=self.data_root, train=False, download=True)
        self.num_classes = len(train.classes)
        self.num_step = len(train) // self.batch_size

        print('-' * 50)
        print('* {} dataset class num: {}'.format(self.dataset_name, len(train.classes)))
        print('* {} train dataset len: {}'.format(self.dataset_name, len(train)))
        print('* {} test dataset len: {}'.format(self.dataset_name, len(test)))
        print('-' * 50)

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            ds = self.dataset(root=self.data_root, train=True, transform=self.train_transform)
            self.train_ds, self.valid_ds = self.split_train_valid(ds)

        elif stage in (None, 'test', 'predict'):
            self.test_ds = self.dataset(root=self.data_root, train=False, transform=self.test_transform)

    def split_train_valid(self, ds):
        ds_len = len(ds)
        valid_ds_len = int(ds_len * self.valid_ratio)
        train_ds_len = ds_len - valid_ds_len
        return random_split(ds, [train_ds_len, valid_ds_len])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class CIFAR(BaseDataModule):
    def __init__(self, dataset_name: str, size: tuple, **kwargs):
        if dataset_name == 'cifar10':
            dataset, mean, std = CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif dataset_name == 'cifar100':
            dataset, mean, std = CIFAR100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)

        train_transform, test_transform = self.get_trasnforms(mean, std, size)
        super(CIFAR, self).__init__(dataset_name, dataset, train_transform, test_transform, **kwargs)

    def get_trasnforms(self, mean, std, size):
        train = transforms.Compose([
            transforms.Resize(size),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return train, test

class BaseVisionSystem(LightningModule):
    def __init__(self, backbone_init: dict, num_classes: int, num_step: int, max_epochs: int,
                 optimizer_init: dict, lr_scheduler_init: dict):
        """ Define base vision classification system
        :arg
            backbone_init: feature extractor
            num_classes: number of class of dataset
            num_step: number of step
            max_epoch: max number of epoch
            optimizer_init: optimizer class path and init args
            lr_scheduler_init: learning rate scheduler class path and init args
        """
        super(BaseVisionSystem, self).__init__()

        self.automatic_optimization = True

        # step 1. save data related info (not defined here)
        self.num_step = num_step
        self.max_epochs = max_epochs

        # step 2. define model
        self.backbone = torch.hub.load('hankyul2/EfficientNetV2-pytorch', **backbone_init)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

        # step 3. define lr tools (optimizer, lr scheduler)
        self.optimizer_init_config = optimizer_init
        self.lr_scheduler_init_config = lr_scheduler_init
        self.criterion = nn.CrossEntropyLoss()

        # step 4. define metric
        metrics = MetricCollection({'top@1': Accuracy(top_k=1, task="multiclass", num_classes=10), 'top@5': Accuracy(top_k=5, task="multiclass", num_classes=10)})
        self.train_metric = metrics.clone(prefix='train/')
        self.valid_metric = metrics.clone(prefix='valid/')
        self.test_metric = metrics.clone(prefix='test/')

    def forward(self, x):
        return self.fc(self.backbone(x))

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, self.train_metric, 'train', add_dataloader_idx=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.valid_metric, 'valid', add_dataloader_idx=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.test_metric, 'test', add_dataloader_idx=True)

    def shared_step(self, batch, metric, mode, add_dataloader_idx):
        x, y = batch
        loss, y_hat = self.compute_loss(x, y) if mode == 'train' else self.compute_loss_eval(x, y)
        metric = metric(y_hat, y)
        self.log_dict({f'{mode}/loss': loss}, add_dataloader_idx=add_dataloader_idx)
        self.log_dict(metric, add_dataloader_idx=add_dataloader_idx, prog_bar=True)
        return loss

    def compute_loss(self, x, y):
        return self.compute_loss_eval(x, y)

    def compute_loss_eval(self, x, y):
        y_hat = self.fc(self.backbone(x))
        loss = self.criterion(y_hat, y)
        return loss, y_hat

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 0.1},
            {'params': self.fc.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {
            'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
            'interval': 'step'
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def update_and_get_lr_scheduler_config(self):
        if 'T_max' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['T_max'] = self.num_step * self.max_epochs
        return self.lr_scheduler_init_config


def update_config(config, data):
    config['model']['num_classes'] = data.num_classes
    config['model']['num_step'] = data.num_step
    config['model']['max_epochs'] = config['trainer']['max_epochs']


if __name__ == '__main__':
    data = CIFAR(**config['data'])
    update_config(config, data)
    model = BaseVisionSystem(**config['model'])
    trainer = Trainer(**config['trainer'])
    trainer.fit(model, data)
    trainer.test(ckpt_path='best')