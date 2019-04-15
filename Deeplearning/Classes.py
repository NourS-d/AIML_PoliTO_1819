from torchvision import models

#################################################################
#########        Multi Layer Perceptron               ###########
#################################################################

class old_nn(nn.Module):
    def __init__(self):
        super(old_nn, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes) #last FC for classification 

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x




#################################################################
#########       Convultional Neural Network           ###########
#################################################################

class CNN(nn.Module):
    def __init__(self,s1=32,s2=32,s3=32,s4=64,o=4096,BN=False,DO=False):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, s1, kernel_size=5, stride=2, padding=0)
        self.conv1_bn = nn.BatchNorm2d(s1)
        self.conv2 = nn.Conv2d(s1, s2, kernel_size=3, stride=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(s2)
        self.conv3 = nn.Conv2d(s2, s3, kernel_size=3, stride=1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(s3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_final = nn.Conv2d(s3, s4, kernel_size=3, stride=1, padding=0)
        self.conv_final_bn = nn.BatchNorm2d(s4)
        self.fc1 = nn.Linear(s4 * 4 * 4, o)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(o, n_classes)
        self.BN=BN
        self.DO=DO
    def forward(self, x):
        if(self.BN==False):
          x = F.relu(self.conv1(x))
          x = F.relu(self.conv2(x))
          x = F.relu(self.conv3(x))
          x = F.relu(self.pool(self.conv_final(x)))
          x = x.view(x.shape[0], -1)
          x = F.relu(self.fc1(x))
        else:
          x = F.relu(self.conv1_bn(self.conv1(x)))
          x = F.relu(self.conv2_bn(self.conv2(x)))
          x = F.relu(self.conv3_bn(self.conv3(x)))
          x = F.relu(self.pool(self.conv_final_bn(self.conv_final(x))))
          x = x.view(x.shape[0], -1)
          x = F.relu(self.fc1(x))
        
        #hint: dropout goes here!
        if(self.DO):
          x = self.dropout(x)
          
        x = self.fc2(x)
        return x

# The one that gives the best accuracy was:
BestCNN = CNN(128,128,128,256,4096,BN=True,DO=True)


#################################################################
#########                   ResNet18                  ###########
#################################################################

#We used the following one:

net = models.resnet18(pretrained=True)
net.fc = nn.Linear(512, n_classes)
