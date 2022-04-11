import torchvision.models as models
# import efficient_net.models as models
import torch
import torch.nn as nn
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # self.net = models.efficientnet_b3(pretrained=True)
        self.net = models.resnet50(pretrained=True)
        # print(list(self.net.named_modules())[0][0][0][:-1])
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Flatten(),
            # nn.Linear(in_features=1536, out_features=1, bias=True)
            nn.Linear(in_features=2048, out_features=1, bias=True)
        # nn.Linear(in_features=1408, out_features=1, bias=True)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        # torch_x = torch.sigmoid(x)
        return x

if __name__ == '__main__':
    model = Net()
    print(model)
    # print(*list(model.children())[:-1])