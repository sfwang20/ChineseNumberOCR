import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np

# Define your net here
class LeNet(nn.Module):
    # define the layers
    def __init__(self):
        super(LeNet, self).__init__()
        print('Building model...')
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        
    # connect these layers
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ChineseOCR(object):
    """docstring for ChineseOCR"""
    def __init__(self, in_path, epoch, batch_size, lr):
        super(ChineseOCR, self).__init__()
        self.in_path = in_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.classes = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

        self.checkdevice()
        self.prepareData()
        
        self.getModel()
        self.train_acc = self.train()
        self.saveModel()
        
        self.loadModel("./models/resnet18.pt")
        
        self.test()
        self.showWeights()

    def checkdevice(self):
        # To determine if your system supports CUDA
        print("Check devices...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Current device:", self.device)

        # Print your current GPU id, and the number of GPUs you can use.
        print("Our selected device:", torch.cuda.current_device())
        print(torch.cuda.device_count(), "GPUs is available")
        return

    def prepareData(self):
        print('Preparing dataset...')

        # The transform function for train/val/test data
        transform_train = transforms.Compose([
            transforms.RandomCrop(128, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])

        self.trainset = torchvision.datasets.ImageFolder("./dataset/train", transform=transform_train)
        self.validset = torchvision.datasets.ImageFolder("./dataset/val", transform=transform_val)
        self.testset = torchvision.datasets.ImageFolder("./dataset/test", transform=transform_test)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=self.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
        return

    def getModel(self):
        self.net = torchvision.models.resnet18(pretrained = True)
        print(self.net)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        #self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        return
        
    def train(self):
        print('Training model...')
        # Change all model tensor into cuda type
        self.net = self.net.to(self.device) 
        
        # Set the model in training mode
        self.net.train()

        for e in range(self.epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            correct = 0
            for i, (inputs, labels) in enumerate(self.trainloader, 0):                
                #change the type into cuda tensor 
                inputs, labels = inputs.to(self.device), labels.to(self.device)                

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)

                # select the class with highest probability   
                _, pred = outputs.max(1)   
                # if the model predicts the same results as the true label, then the correct counter will plus 1
                correct += pred.eq(labels).sum().item()
                
                loss = self.criterion(outputs, labels)
                
                loss.backward()         # computes dloss/dx for every parameter x which has requires_grad=True
                self.optimizer.step()   # updates the value using the gradient

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print(f"[{e+1}, {i+1}] loss: {round(running_loss/100, 3)}")
                    running_loss = 0.0
            accuracy = 100. * correct/len(self.trainset)
            print(f"{e+1} epoch, training accuracy: {round(accuracy, 4)}")

            # validation
            running_loss = 0.0
            val_correct = 0
            iter_count = 0
            class_correct = [0 for i in range(len(self.classes))]
            class_total = [0 for i in range(len(self.classes))]
            with torch.no_grad():
                for data in self.validloader:                
                    images, labels = data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.net(images)
                    _, pred = outputs.max(1)               
                    val_correct += pred.eq(labels).sum().item()  
                    #c_eachlabel = pred.eq(labels).squeeze()  
                    loss = self.criterion(outputs, labels)
                    iter_count += 1
                    running_loss += loss.item()
            # print accuracy and loss
            train_accuracy = 100. * correct/len(self.trainset)
            val_accuracy = 100. * val_correct/len(self.validset)
            print(f"{e+1} epoch, training accuracy: {round(train_accuracy, 3)}, val accuracy: {round(val_accuracy, 3)}, val loss {round(running_loss/iter_count, 3)}")
            running_loss = 0.0
    
        print('Finished Training')
        return 100.*correct/len(self.trainset)

    def test(self):
        print(self.net)
        # If only testing, need to define the criterion
        self.criterion = nn.CrossEntropyLoss()     

        print('==> Testing model..')
        # Change model to cuda tensor
        # or it will raise when images and labels are all cuda tensor type
        self.net = self.net.to(self.device)

        # Set the model in evaluation mode
        self.net.eval()

        correct = 0
        running_loss = 0.0
        iter_count = 0
        class_correct = [0 for i in range(len(self.classes))]
        class_total = [0 for i in range(len(self.classes))]
        with torch.no_grad(): # no need to keep the gradient for backpropagation
            for images, labels in self.testloader:     
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, pred = outputs.max(1) 
                correct += pred.eq(labels).sum().item()
                c_eachlabel = pred.eq(labels).squeeze()
                loss = self.criterion(outputs, labels)
                iter_count += 1
                running_loss += loss.item()

                for i in range(len(labels)):
                    cur_label = labels[i].item()
                    try:
                        class_correct[cur_label] += c_eachlabel[i].item()
                    except:
                        print(class_correct[cur_label])
                        print(c_eachlabel[i].item())
                    class_total[cur_label] += 1

        print('Total accuracy is: {:4f}% and loss is: {:3.3f}'.format(100 * correct/len(self.testset), running_loss/iter_count))
        print('For each class in dataset:')
        for i in range(len(self.classes)):
            print('Accruacy for {:18s}: {:4.2f}%'.format(self.classes[i], 100 * class_correct[i]/class_total[i]))

    def saveModel(self):
        # After training , save the model first
        # we can save only the model parameters or entire model

        print('Saving model...')
        # only save model parameters
        torch.save(self.net.state_dict(), './weight.t7')

        # we also can store some log information
        state = {
            'net': self.net.state_dict(),
            'acc': self.train_acc,
            'epoch': self.epoch
        }
        torch.save(state, './weight.t7')

        # save entire model
        torch.save(self.net, './model.pt')
        return

    def loadModel(self, path):
        print('Loading model...')        
        if path.split('.')[-1] == 't7':
            # If you just save the model parameters, you
            # need to redefine the model architecture, and
            # load the parameters into your model
            self.net = LeNet()
            checkpoint = torch.load(path)
            self.net.load_state_dict(checkpoint['net'])
        elif path.split('.')[-1] == 'pt':
            # If you save the entire model
            self.net = torch.load(path)
        return

    def showWeights(self):
        # layer 1
        w_layer1_block0_conv1 = self.net.layer1[0].conv1.weight.reshape(-1).detach().cpu().numpy()
        w_layer1_block0_conv2 = self.net.layer1[0].conv2.weight.reshape(-1).detach().cpu().numpy()
        w_layer1_block1_conv1 = self.net.layer1[1].conv1.weight.reshape(-1).detach().cpu().numpy()
        w_layer1_block1_conv2 = self.net.layer1[1].conv2.weight.reshape(-1).detach().cpu().numpy()

        # layer 2
        w_layer2_block0_conv1 = self.net.layer2[0].conv1.weight.reshape(-1).detach().cpu().numpy()
        w_layer2_block0_conv2 = self.net.layer2[0].conv2.weight.reshape(-1).detach().cpu().numpy()
        w_layer2_block1_conv1 = self.net.layer2[1].conv1.weight.reshape(-1).detach().cpu().numpy()
        w_layer2_block1_conv2 = self.net.layer2[1].conv2.weight.reshape(-1).detach().cpu().numpy()

        # layer 3
        w_layer3_block0_conv1 = self.net.layer3[0].conv1.weight.reshape(-1).detach().cpu().numpy()
        w_layer3_block0_conv2 = self.net.layer3[0].conv2.weight.reshape(-1).detach().cpu().numpy()
        w_layer3_block1_conv1 = self.net.layer3[1].conv1.weight.reshape(-1).detach().cpu().numpy()
        w_layer3_block1_conv2 = self.net.layer3[1].conv2.weight.reshape(-1).detach().cpu().numpy()

        # layer 4
        w_layer4_block0_conv1 = self.net.layer4[0].conv1.weight.reshape(-1).detach().cpu().numpy()
        w_layer4_block0_conv2 = self.net.layer4[0].conv2.weight.reshape(-1).detach().cpu().numpy()
        w_layer4_block1_conv1 = self.net.layer4[1].conv1.weight.reshape(-1).detach().cpu().numpy()
        w_layer4_block1_conv2 = self.net.layer4[1].conv2.weight.reshape(-1).detach().cpu().numpy()

        # FC
        w_fc = self.net.fc.weight.reshape(-1).detach().cpu().numpy()

        print("Ploting the weights figure...")

        plt.figure(figsize=(28, 16))
        plt.subplot(3,6,1)
        plt.title("L1 b0 conv1")
        plt.hist(w_layer1_block0_conv1)

        plt.subplot(3,6,2)
        plt.title("L1 b0 conv2")
        plt.hist(w_layer1_block0_conv2)

        plt.subplot(3,6,3)
        plt.title("L1 b1 conv1")
        plt.hist(w_layer1_block1_conv1)

        plt.subplot(3,6,4)
        plt.title("L1 b1 conv2")
        plt.hist(w_layer1_block1_conv2)
        # layer2
        plt.subplot(3,6,5)
        plt.title("L2 b0 conv1")
        plt.hist(w_layer2_block0_conv1)

        plt.subplot(3,6,6)
        plt.title("L2 b0 conv2")
        plt.hist(w_layer2_block0_conv2)

        plt.subplot(3,6,7)
        plt.title("L2 b1 conv1")
        plt.hist(w_layer2_block1_conv1)

        plt.subplot(3,6,8)
        plt.title("L2 b1 conv2")
        plt.hist(w_layer2_block1_conv2)
        # layer 3
        plt.subplot(3,6,9)
        plt.title("L3 b0 conv1")
        plt.hist(w_layer3_block0_conv1)

        plt.subplot(3,6,10)
        plt.title("L3 b0 conv2")
        plt.hist(w_layer3_block0_conv2)

        plt.subplot(3,6,11)
        plt.title("L3 b1 conv1")
        plt.hist(w_layer3_block1_conv1)

        plt.subplot(3,6,12)
        plt.title("L3 b1 conv2")
        plt.hist(w_layer3_block1_conv2)
        # layer 4
        plt.subplot(3,6,13)
        plt.title("L4 b0 conv1")
        plt.hist(w_layer4_block0_conv1)

        plt.subplot(3,6,14)
        plt.title("L4 b0 conv2")
        plt.hist(w_layer4_block0_conv2)

        plt.subplot(3,6,15)
        plt.title("L4 b1 conv1")
        plt.hist(w_layer4_block1_conv1)

        plt.subplot(3,6,16)
        plt.title("L4 b1 conv2")
        plt.hist(w_layer4_block1_conv2)
        # FC
        plt.subplot(3,6,17)
        plt.title("FC")
        plt.hist(w_fc)

        plt.savefig('weights.png', bbox_inches='tight')
        print("Done and saved!")

        
if __name__ == '__main__':
    ocr = ChineseOCR('./data', 95, 26, 0.001)