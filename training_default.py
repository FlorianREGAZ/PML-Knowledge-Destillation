import logging

import torch
import torch.nn as nn
import torch.optim as optim
import timm


from utils import train, evaluate, get_device, get_dataset_loader

torch.manual_seed(0)

device = get_device()
logging.info(f'Using device: {device}')

trainloader, testloader = get_dataset_loader()

# Model: GhostNetV3 pretrained
model = timm.create_model('ghostnetv3_1.0', pretrained=True, num_classes=10)
model.to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 100
best_acc = 0.0
for epoch in range(1, num_epochs + 1):
    train(model, device, trainloader, criterion, optimizer, epoch)
    acc = evaluate(model, device, testloader, criterion)
    scheduler.step()
    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_ghostnetv3_cifar10.pth')
        logging.info(f'New best accuracy: {best_acc:.2f}%, model saved.')

logging.info(f'Training complete. Best Test Accuracy: {best_acc:.2f}%')
