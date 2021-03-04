import torch.nn as nn

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
