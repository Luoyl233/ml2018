import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os


###############################define CNN######################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=0)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, padding=0)
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1)
        self.glb_avg = nn.AdaptiveAvgPool2d(1)
        #self.glb_avg = nn.AvgPool2d(6)

    def forward(self, x):
        x_drop = F.dropout(x, 0.2)
        x_conv1 = F.relu(self.conv1(x_drop))
        x_conv2 = F.relu(self.conv2(x_conv1))
        x_conv3 = F.relu(self.conv3(x_conv2))
        x_drop = F.dropout(x_conv3, 0.5)
        x_conv4 = F.relu(self.conv4(x_drop))
        x_conv5 = F.relu(self.conv5(x_conv4))
        x_conv6 = F.relu(self.conv6(x_conv5))
        x_drop = F.dropout(x_conv6, 0.5)
        x_conv7 = F.relu(self.conv7(x_drop))
        x_conv8 = F.relu(self.conv8(x_conv7))
        class_out = F.relu(self.conv9(x_conv8))
        pool_out = self.glb_avg(class_out)
        #out = pool_out.view(-1, 10)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)

        return pool_out


###############################load data######################################
def load_data(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


if __name__ == '__main__':
    batch_size = 256
    trainloader, testloader, classes = load_data(batch_size)

    load_net_from_file = False
    path = './'
    load_file_name = path + 'cnn2_epoch_24.nn'
    max_epoch = 30
    train_loss = np.zeros((max_epoch, 1))
    train_acc_rate = np.zeros((max_epoch, 1))

    net = Net()
    if torch.cuda.is_available():
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)

    ###############################begin trainning######################################
    for epoch in range(max_epoch):  # 循环遍历数据集多次
        start = time.clock()
        running_loss = 0.0
        running_loss_per_epoch = 0.0
        correct = 0
        total = 0

        # 直接从文件中加载
        if load_net_from_file == True:
            break

        for i, data in enumerate(trainloader, 0):
            # 得到输入数据
            inputs, labels = data

            # 包装数据
            #inputs, labels = Variable(inputs), Variable(labels)
            #use GPU
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)


            # 梯度清零
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印信息
            running_loss += loss.item()
            running_loss_per_epoch += loss.item()
            if i % 500 == 499:    # 每500个小批量打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        #for safe, save model for every epoch
        print('saving net to file...')
        torch.save(net.state_dict(), path+'cnn2_epoch_' + str(epoch+1) + '.nn')

        # 统计每个epoch信息
        end = time.clock()
        train_loss[epoch] = running_loss_per_epoch * batch_size / 50000
        train_acc_rate[epoch] = 100 * correct / total
        print("epoch %d finished in %d seconds: loss=%f, acc_rate=%f %%"
              % (epoch + 1, (end-start), train_loss[epoch], train_acc_rate[epoch]))

        # save loss, acc_rate to file
        np.savetxt(path + "cnn2_train_loss_" + str(epoch + 1) + ".txt", train_loss)
        np.savetxt(path + "cnn2_train_acc_rate_" + str(epoch + 1) + ".txt", train_acc_rate)

    if load_net_from_file == False:
        print('Finished Training')
        print('saving net to file...')
        torch.save(net.state_dict(), 'cnn2_epoch_'+str(max_epoch)+'.nn')
    else:
        print('loading net form file...')
        net.load_state_dict(torch.load(load_file_name))

    ###############################begin testing######################################
    print('begin testing')
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)

        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        if class_total[i] == 0:
            print('Accuracy of %5s : %2f %%' % (
                classes[i], 0))
        else:
            print('Accuracy of %5s : %2f %%' % (
                classes[i], 100.0 * float(class_correct[i]) / float(class_total[i])))

    print('Accuracy of the network on the 10000 test images: %f %%' % (
            100.0 * correct / total))

    #绘制图像
    x = np.linspace(1, max_epoch, max_epoch)
    plt.plot(x, train_loss, 'r-o', label='Train loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Waves')
    plt.savefig(path + "cnn2_train_loss_" + str(max_epoch) + ".jpg")
    plt.figure()
    plt.plot(x, train_acc_rate, 'g-o', label='Train acc rate')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc rate')
    plt.title('acc rate Waves')
    plt.savefig(path + "cnn2_train_acc_rate_" + str(max_epoch) + ".jpg")
    plt.show()