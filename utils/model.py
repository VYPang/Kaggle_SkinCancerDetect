import torch.nn as nn
import torch.nn.functional as f

class basicCNN(nn.Module):
    def __init__(self, shape, num_class, test=False):
        super(basicCNN, self).__init__()
        self.test = test
        channel = shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        if self.test:
            vector = x
        x = self.output(x)
        if self.test:
            return f.log_softmax(x, dim=1), vector
        return f.log_softmax(x, dim=1)