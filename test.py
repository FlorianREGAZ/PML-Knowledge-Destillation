import logging
import timm
import torch
import torch.nn as nn

import models.ghostnetv3 as ghostnetv3
import models.ghostnetv3_small as ghostnetv3_small
from models.resnet import resnet50, resnet34, resnet18
from models.densenet import densenet121, densenet161, densenet169
from models.inception import inception_v3
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

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
    _, testloader = get_dataset_loader()

    #model = resnet18(pretrained=True, device=device).to(device)
    model = timm.create_model('ghostnetv3_small', width=2.8, num_classes=10)
    checkpoint = torch.load("default_ghostnetv3_small_cifar10.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print(sum(p.numel() for p in model.parameters()))

    acc = evaluate(model, device, testloader, nn.CrossEntropyLoss())

    logging.info(f'Test complete. Test Accuracy: {acc:.2f}%')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
