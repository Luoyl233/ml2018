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
import simple_cnn


if __name__ == '__main__':
    train_batch_size = 4
    test_batch_size = 4
    trainloader, testloader, classes = simple_cnn.load_data(train_batch_size, test_batch_size)
    path = './'
    max_epoch = 1
    net = simple_cnn.Net()
    if torch.cuda.is_available():
        net.cuda()
    epoch_acc_rate = np.zeros((max_epoch, 1))
    class_acc_rate = np.zeros((10, max_epoch))

    for epoch in range(max_epoch):
        print('-'*30, 'test epoch ' + str(epoch+1), '-'*30)
        net.load_state_dict(torch.load(path + 'simple_cnn_epoch_' + str(epoch+1) + '.nn'))
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
                class_acc_rate[i][epoch] = 100.0 * float(class_correct[i]) / float(class_total[i])

        print('Accuracy of the network on the 10000 test images: %f %%' % (
                100.0 * correct / total))
        epoch_acc_rate[epoch] = 100.0 * correct / total


        # 绘制图像
    x = np.linspace(1, max_epoch, max_epoch)
    plt.plot(x, epoch_acc_rate, 'r-o', label='Test acc rate')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc rate')
    plt.title('acc rate Waves')
    plt.savefig(path + "simple_cnn_test_acc_rate_" + str(max_epoch) + ".jpg")

    plt.figure()
    row = 1
    col = 1
    color = ['#FF0000', '#FF8000', '#FFFF00', '#80FF00', '#00FF00',
             '#00FF80', '#00FFFF', '#0080FF', '#0000FF', '#8000FF']
    for i in range(10):
        if i == 5:
            row += 1
            col = 1
        #plt.subplot(row,col,1)
        col += 1
        plt.plot(x, class_acc_rate[i], color=color[i],linestyle='-', marker='o', label=classes[i])
        plt.legend()
        plt.legend
    plt.xlabel('epoch')
    plt.ylabel('acc rate')
    plt.title('acc rate Waves')
    plt.savefig(path + "simple_cnn_class_acc_rate_" + str(max_epoch) + ".jpg")
    plt.show()
