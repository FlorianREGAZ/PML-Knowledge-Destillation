import logging
import timm
import torch
import torch.nn as nn

from train_efficientnetv2 import BaseVisionSystem, config
from utils import (
    evaluate,
    get_device,
    get_dataset_loader,
)

def main():
    print("Starting Test.")
    torch.manual_seed(0)
    device = get_device()
    logging.info(f'Using device: {device}')
    _, testloader = get_dataset_loader(resize=(224, 224))

    config['model']['num_classes'] = 10
    config['model']['num_step'] = 150000
    config['model']['max_epochs'] = 100
    #teacher = BaseVisionSystem(**config['model'])
    model = BaseVisionSystem.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=99-step=140700.ckpt", **config['model'])
    model.eval()
    model.to(device)
    print(sum(p.numel() for p in model.parameters()))

    acc = evaluate(model, device, testloader, nn.CrossEntropyLoss())

    logging.info(f'Test complete. Test Accuracy: {acc:.2f}%')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
