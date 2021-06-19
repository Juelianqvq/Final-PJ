import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet50,densenet201,resnet18
from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter()
cifar_10_dir = r'D:\Project-Torch\cifar-10-batches-py'

transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=cifar_10_dir, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root=cifar_10_dir, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = resnet18()
#model=densenet201()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

loss=nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,70],gamma=0.1)
# warmup_epoch = 4
#
# warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warmup_epoch if epoch < warmup_epoch \
#     else 0.5 * (np.cos((epoch - warmup_epoch) / (num_epoch - warmup_epoch) * np.pi) + 1)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
num_epoch=80

def train():

    for epoch in range(num_epoch):
        train_loss=0

        for i,data in enumerate(trainloader):
            input,label=data
            input,label=input.to(device),label.to(device)
            optimizer.zero_grad()
            outputs = model(input)
            batch_loss = loss(outputs, label.view(-1))
            batch_loss.backward()
            optimizer.step()
            lr_schedule.step()
            #scheduler.step()
            train_loss += batch_loss.item()

            if i % 200 == 199:
                print('epoch:%d|batch:%d|loss:%.3f' % (epoch+1,i+1,train_loss/200))
                train_loss = 0.0

        writer.add_scalar('loss_curve',batch_loss,epoch)
        writer.add_graph(model,input)


        correct=0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            print('Accuracy : %.2f%%' % (correct / 100))

            print('---------------------')

        writer.add_scalar('accuracy',correct/100,epoch)

if __name__=='__main__':
        train()
