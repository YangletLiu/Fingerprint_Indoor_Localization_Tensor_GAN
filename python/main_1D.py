# coding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as scio
import numpy as np

# Expected data structure:
# 12500*5*[1*169,count]
# (batches)*(number of matrix in batches)*(size of matrix and label)


# Data PreProcess
with open('Radio_map_3D.mat','rb') as fileholder:
    trainset = scio.loadmat(fileholder)
    v = np.array(trainset['Radio_map_3D'])

print('Data read complete')
labeled_trainset = []
tensor = []
# 将250*250*169的tensor转换成62500*169的tensor
for row in v:
    for vector in row:
        tensor.append([vector])
trainset = torch.FloatTensor(tensor)
for i,j in enumerate(trainset,0):
    labeled_trainset.append((j,i))

trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=5,shuffle=True, num_workers=16)

print('Data load finished')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 5)
        self.pool  = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 39, 6250)
        self.fc2   = nn.Linear(6250, 31250)
        self.fc3   = nn.Linear(31250, 62500)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print('layer 1')
        #print(x)
        x = self.pool(F.relu(self.conv2(x)))
        print('layer 2')
        #print(x)
        x = x.view(-1, 16 * 39)
        print('layer 3')
        #print(x)
        x = F.relu(self.fc1(x))
        print('layer 4')
        #print(x)
        x = F.relu(self.fc2(x))
        print('layer 5')
        #print(x)
        x = self.fc3(x)
        print('layer 6')
        #print(x)
        return x

net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

'''
# 以下代码是为了看一下我们需要训练的参数的数量
print(net)
params = list(net.parameters())

k=0
for i in params:
    l =1
    print("该层的结构："+str(list(i.size())))
    for j in i.size():
        l *= j
    print("参数和："+str(l))
    k = k+l

print("总参数和："+ str(k))
'''

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):# 0 means the start of index
        # get the inputs
        inputs, labels = data
        #labels = torch.LongTensor([40,9,8,0,0])
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        print(inputs)
        print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2500 == 2499:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
