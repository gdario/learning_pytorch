import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

use_gpu = True
NUM_EPOCHS = 25
data_dir = 'hymenoptera_data'
subfolders = ['train', 'val']

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def image_folder(datadir, subdir, transformations):
    out = datasets.ImageFolder(os.path.join(datadir, subdir),
                               transformations[subdir])
    return out


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


image_datasets = {
    x: image_folder(data_dir, x, data_transforms)
    # x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in subfolders
}

data_loaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
    for x in subfolders
}

dataset_sizes = {x: len(image_datasets[x]) for x in subfolders}
class_names = image_datasets['train'].classes


def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in subfolders:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_num_correct = 0

            for data in data_loaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                # noinspection PyArgumentList
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0] * inputs.size(0)
                running_num_correct += torch.sum(preds.data == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_num_correct / dataset_sizes[phase]
            # import ipdb; ipdb.set_trace() # DEBUG
            print('{} Loss: {:.4f} A: {:.4f}'.format(
                phase, epoch_loss, epoch_accuracy))

            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:4f}'.format(best_accuracy))

    model.load_state_dict(best_model_weights)
    return model


def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(data_loaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        # noinspection PyArgumentList
        _, preds = torch.max(outputs.data, dim=1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


model_fine_tuning = models.resnet18(pretrained=True)
num_features = model_fine_tuning.fc.in_features
model_fine_tuning.fc = nn.Linear(num_features, 2)

if use_gpu:
    model_fine_tuning = model_fine_tuning.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_fine_tuning = optim.SGD(model_fine_tuning.parameters(), lr=0.001,
                                  momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_fine_tuning, step_size=7, gamma=0.1)

model_fine_tuning = train_model(model_fine_tuning, criterion,
                                optimizer_fine_tuning, exp_lr_scheduler)

visualize_model(model_fine_tuning)
