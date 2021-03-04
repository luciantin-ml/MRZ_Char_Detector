import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch import optim
from torch.utils.data import random_split

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)


in_x = 88
in_y = 57
# classes = len(digits_index_to_char)
# print(classes)  # 37

in_x = 28
in_y = 28
classes = 10
print(classes)  # 37


char_model = nn.Sequential(
    nn.Linear(in_x*in_y, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, classes)
)


# optimizer


optimiser = optim.SGD(char_model.parameters(), lr=1e-2)

# loss

loss = nn.CrossEntropyLoss()

# training loop
acc = list()

epochs = 5

for epoch in range(0, epochs):
    losses = list()
    char_model.train()
    for batch in train_loader:
        x, y = batch

        b = x.size(0)
        x = x.view(b, -1)

        # 1 forward
        l = char_model(x)  # l: logits

        # 2 compute the objective function
        J = loss(l, y)

        # 3 cleaning the gradients
        char_model.zero_grad()

        # 4 accumulate the partial derivatives of J with respect to params
        J.backward()

        # 5 step in the opposite direction of the gradient
        optimiser.step()

        losses.append(J.item())

        acc.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}, train loss : {torch.tensor(losses).mean():.2f}')
    print(f'Epoch {epoch + 1}, train acc : {torch.tensor(acc).mean():.2f}')

