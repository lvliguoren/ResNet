import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import torch
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.tensorboard import SummaryWriter


# 预训练的参数
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(inplanes,planes,stride=1):
    return nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False)

# 基础残差单元
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[1],stride=2)
        self.layer4 = self._make_layer(block,512,layers[1],stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if(stride!=1 or self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,stride, downsample))
        self.inplanes =  planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



#对输入图像进行处理，转换为（224，224）,因为resnet18要求输入为（224，224），并转化为tensor
def input_transform():
    return Compose([
                Resize(224),   #改变尺寸
                ToTensor(),      #变成tensor
                ])


train_data = torchvision.datasets.MNIST(
    root='./data/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=input_transform(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=False,          # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.MNIST(
    root='./data/',    # 保存或者提取位置
    train=False,  # this is training data
    transform=input_transform(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=False,          # 没下载就下载, 下载了就不用再下了
)


'''
进行批处理
'''
train_loader = Data.DataLoader(dataset=train_data,
                        batch_size=128,
                        shuffle=True)

test_loader = Data.DataLoader(dataset=test_data,
                        batch_size=128,
                        shuffle=True)

net = resnet18(num_classes=10)
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()


def train(train_x, train_y):
    net.train()

    # 前向传播
    train_outputs = net(train_x)
    loss = loss_func(train_outputs, train_y)

    # 后向传播及梯度优化
    # 梯度置零，不然每个batch都会叠加
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算准确率
    _, argmax = torch.max(train_outputs, 1)
    accuracy = (train_y == argmax.squeeze()).float().mean()

    return loss.item(), accuracy.item()

@torch.no_grad()
def test():
    net.eval()

    total = 0
    correct = 0
    for test_x, test_y in test_loader:
        test_outputs = net(test_x)
        _, argmax = torch.max(test_outputs, 1)
        correct += (test_y == argmax.squeeze()).sum().item()
        total += test_y.size(0)
        # 只算第一个批次不然太慢了
        break
    return correct / total

writer = SummaryWriter()

for epoch in range(2):
    for step,(batch_x,batch_y) in enumerate(train_loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        train_loss, train_accuracy = train(b_x, b_y)

        if step % 5 == 0:
            test_accuracy = test()
            print('epoch:{}, step:{}, train_loss:{}, train_accuracy:{}, test_accuracy:{}'.format(epoch, step, train_loss, train_accuracy, test_accuracy))

            if(step == 5):
                writer.add_graph(net, b_x)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalars('Accuracy', {"train":train_accuracy,"test":test_accuracy}, epoch)

writer.close()
torch.save(net,"model.pth")