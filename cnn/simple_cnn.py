import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from math import log


###############################define CNN######################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5  # 非标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_data(train_batch_size, test_batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


if __name__ == '__main__':
    train_batch_size = 4
    test_batch_size = 4
    trainloader, testloader, classes = load_data(train_batch_size, test_batch_size)

    load_net_from_file = False
    path = './'
    load_file_name = path + 'cnn1_epoch_12.nn'
    max_epoch = 30
    train_loss = np.zeros((max_epoch, 1))
    train_acc_rate = np.zeros((max_epoch, 1))

    net = Net()
    if torch.cuda.is_available():
        net.cuda()

    lr = 0.0005 * log(train_batch_size, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
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
            # use GPU
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

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            running_loss_per_epoch += loss.item()
            # 打印信息
            running_loss += loss.item()
            if i % 500 == 499:  # 每2000个小批量打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

        # 统计每个epoch信息
        end = time.clock()
        train_loss[epoch] = running_loss_per_epoch * train_batch_size / 50000
        train_acc_rate[epoch] = 100 * correct / total
        print("epoch %d finished in %d seconds: loss=%f, acc_rate=%f %%"
              % (epoch + 1, (end - start), train_loss[epoch], train_acc_rate[epoch]))

        #save loss, acc_rate to file
        np.savetxt(path+"cnn1_train_loss_"+str(epoch+1)+".txt", train_loss)
        np.savetxt(path + "cnn1_train_acc_rate_" + str(epoch+1) + ".txt", train_acc_rate)

        #for safe, save model for every epoch
        print('saving net to file...')
        torch.save(net.state_dict(), path+'cnn1_epoch_' + str(epoch+1) + '.nn')

    if load_net_from_file == False:
        print('Finished Training')
        print('saving net to file...')
        torch.save(net.state_dict(), path + 'cnn1_epoch_' + str(max_epoch) + '.nn')
    else:
        print('loading net form file...')
        net.load_state_dict(torch.load(load_file_name))

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    ###############################begin testing######################################
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
        for i in range(test_batch_size):
            # the size of last group may less than test_batch_size
            if i >= labels.size(0):
                break
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        if class_total[i] == 0:
            print('Accuracy of %5s : %f %%' % (
                classes[i], 0))
        else:
            print('Accuracy of %5s : %f %%' % (
                classes[i], 100.0 * float(class_correct[i]) / float(class_total[i])))

    print('Accuracy of the network on the 10000 test images: %f %%' % (
            100.0 * correct / total))

    #绘制图像
    #train_loss = np.loadtxt(path+"train_loss_"+str(i+1)+".txt")
    #train_acc_rate = np.loadtxt(path + "train_acc_rate_" + str(i+1) + ".txt")
    x = np.linspace(1, max_epoch, max_epoch)
    plt.plot(x, train_loss, 'r-o', label='Train loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Waves')
    plt.savefig(path + "cnn1_train_loss_" + str(max_epoch) + ".jpg")
    plt.figure()
    plt.plot(x, train_acc_rate, 'g-o', label='Train acc rate')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc rate')
    plt.title('acc rate Waves')
    plt.savefig(path + "cnn1_train_acc_rate_" + str(max_epoch) + ".jpg")
    plt.show()