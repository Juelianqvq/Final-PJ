import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from module import ConvAttention, PreNorm, FeedForward
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np


writer = SummaryWriter()
cifar_10_dir = r'D:\Project-Torch\cifar-10-batches-py'

transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



trainset = torchvision.datasets.CIFAR10(root=cifar_10_dir, train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,shuffle=True)

testset = torchvision.datasets.CIFAR10(root=cifar_10_dir, train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=False)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CvT(nn.Module):
    def __init__(self, image_size, in_channels, num_classes, dim=64, kernels=[7, 3, 3], strides=[4, 2, 2],
                 heads=[1, 3, 6] , depth = [1, 2, 10], pool='cls', dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim

        ##### Stage 1 #######
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[0], strides[0], 2),
            Rearrange('b c h w -> b (h w) c', h = image_size//4, w = image_size//4),
            nn.LayerNorm(dim)
        )
        self.stage1_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size//4,depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size//4, w = image_size//4)
        )

        ##### Stage 2 #######
        in_channels = dim
        scale = heads[1]//heads[0]
        dim = scale*dim
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
            Rearrange('b c h w -> b (h w) c', h = image_size//8, w = image_size//8),
            nn.LayerNorm(dim)
        )
        self.stage2_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size//8, depth=depth[1], heads=heads[1], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size//8, w = image_size//8)
        )

        ##### Stage 3 #######
        in_channels = dim
        scale = heads[2] // heads[1]
        dim = scale * dim
        self.stage3_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[2], strides[2], 1),
            Rearrange('b c h w -> b (h w) c', h = image_size//16, w = image_size//16),
            nn.LayerNorm(dim)
        )
        self.stage3_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size//16, depth=depth[2], heads=heads[2], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout, last_stage=True),
        )


        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):

        xs = self.stage1_conv_embed(img)
        xs = self.stage1_transformer(xs)

        xs = self.stage2_conv_embed(xs)
        xs = self.stage2_transformer(xs)

        xs = self.stage3_conv_embed(xs)
        b, n, _ = xs.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        xs = torch.cat((cls_tokens, xs), dim=1)
        xs = self.stage3_transformer(xs)
        xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]

        xs = self.mlp_head(xs)
        return xs


def train(model, device):

    num_epoch=80
    warmup_epoch=4

    warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warmup_epoch if epoch < warmup_epoch \
        else 0.5 * (np.cos((epoch - warmup_epoch) / (num_epoch - warmup_epoch) * np.pi) + 1)

    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):  # loop over the dataset multiple times

        train_loss = 0.0

        for i, data in enumerate(trainloader):
            input,label = data
            input, label = input.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(input)
            batch_loss = loss(output, label)
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += batch_loss.item()

            if i % 200 == 199:
                print('epoch:%d batch:%d loss:%.3f' % (epoch + 1, i + 1, train_loss / 200))
                train_loss = 0.0

        writer.add_scalar('loss_curve', batch_loss, epoch)
        writer.add_scalar('lr',scheduler.get_lr()[0],epoch)
        writer.add_graph(model, input)

        correct = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            print('Accuracy : %.3f%%' % ( correct / 100))

            print('---------------------')

        writer.add_scalar('accuracy', correct / 100, epoch)

if __name__=='__main__':

    model = CvT(32, 3, 10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train(model,device)