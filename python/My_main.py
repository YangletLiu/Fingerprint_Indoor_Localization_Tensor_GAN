import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def pre_process(flag,set):
    for i in enumerate(range(2500),0):
        index,_ = i
        if flag == 0:
            im = Image.open('./train/' + 'cat.' + str(index) + '.jpg' )
        elif flag == 1:
            im = Image.open('./train/' + 'dog.' + str(index) + '.jpg' )
        else:
            im = Image.open('./test/' + str(index + 1) + '.jpg' )
        trans = torchvision.transforms.ToTensor()
        scale = torchvision.transforms.Scale(32)
        center = torchvision.transforms.CenterCrop((32,32))
        im = center(im)
        im = scale(im)
        tensor = trans(im)
        tensor = (tensor,flag)
        set.append(tensor)
        if index % 2500 == 2499:
            print(str(index + 1) + ' processed')

trainset = []
testset = []
pre_process(0,trainset)
pre_process(1,trainset)
#pre_process(2,testset) # flag means test data

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=16)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=16)

classes = ('cat', 'dog')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):# 0 means the start of index
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

count = 1
with open('result.txt', 'w') as f:
    f.writelines('id,label \n')
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        for j in range(4):
            f.writelines(str(count) + ',' + str(predicted[j][0]) + '\n')
            count += 1
print('Finished Testing')