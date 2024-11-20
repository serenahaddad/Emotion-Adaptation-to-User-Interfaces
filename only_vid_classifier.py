#Author: Syrine HADDAD
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import torch.nn.init as init
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers with He initialization
        self.conv1 = nn.Conv2d(3, 32, 5)
        init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        
        # Fully connected layers with Xavier initialization
        self.fc1 = nn.Linear(128 * 30 * 62, 120)
        init.xavier_normal_(self.fc1.weight)
        
        self.fc2 = nn.Linear(120, 84)
        init.xavier_normal_(self.fc2.weight)
        
        self.fc3 = nn.Linear(84, 4)
        init.xavier_normal_(self.fc3.weight)
        
        self.avgpool = nn.AdaptiveAvgPool2d((30, 62))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def print_model_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f'Layer: {name}, Shape: {param.shape}')



def train(data_loader, criterion, net, device, optimizer):
    train_loss = []
    val_loss = []
    val_accuracy = []
    train_accuracy = []

    for epoch in range(25):  
        running_loss = 0.0
        t_loss = 0
        total = 0
        total = 0
        correct = 0
        for i, data in enumerate(data_loader['vid_train']):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            

            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.item()
            t_loss += loss.item()
            if i % 2 == 1:    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0
         
        train_loss.append(t_loss)
        train_accuracy.append(100*correct/total)
        
        correct = 0
        total = 0
        v_loss = 0
        val_minibatch_count = 0
        for i, data in enumerate(data_loader['vid_val']):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            v_loss += loss.item()
            val_minibatch_count += 1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('[Epoch %d, Mini-Batch %5d] Validation Loss: %.3f' % (epoch + 1, val_minibatch_count, loss.item()))
        val_loss.append(v_loss)
        val_accuracy.append(100*correct/total)
        print('Accuracy of the network on the validation dataset: %d %%' % (
                100 * correct / total))

    np.save('train_accuracy_only_video2', train_accuracy)
    np.save('train_loss_only_video2', train_loss)
    np.save('test_accuracy_only_video2', val_accuracy)
    np.save('test_loss_only_video2', val_loss)

    print('Finished Training')
    plt.figure(figsize=(10, 5))  
    plt.plot(train_loss, 'g', label='Training loss')
    plt.plot(val_loss, 'b', label='Validation loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Training_validation_loss2.png')
    plt.show()

    # Plot and save Training and Validation Accuracy
    plt.figure(figsize=(10, 5))  
    plt.plot(train_accuracy, 'g', label='Training Accuracy')
    plt.plot(val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Training_validation_Accuracy2.png')
    plt.show()

def get_lr_scheduler(optimizer, step_size, gamma):
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
def plot_confusion_matrix(confusion_matrix, class_names):
    
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = confusion_matrix / row_sums

    fig, ax = plt.subplots()
    im = ax.imshow(normalized_matrix, cmap='Pastel1')

    
    cbar = ax.figure.colorbar(im, ax=ax)

   
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, f'{normalized_matrix[i, j]:.2f}', ha="center", va="center", color="black")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('ConfusionMatrix2.png')
    plt.show()

def test(data_loader, criterion, net, device, optimizer):
    
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(4, 4)
    
    for epoch in range(25):
        with torch.no_grad():
            for data in data_loader['vid_test']:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                
    class_names = ['Happy', 'Surprise','Calm','Angry']
    print('Confusion Matrix:')
    for row in confusion_matrix:
        print('\t'.join([str(int(value)) for value in row]))
    np.save('confusion_matrix_only_video', confusion_matrix)
    plot_confusion_matrix(confusion_matrix, class_names)
    torch.save(net.state_dict(), 'EmotionGUImodel.pth')           
    np.save('confusion_matrix_only_video2', confusion_matrix)
    print('Accuracy of the network on test set: %d %%' % (
            100 * correct / total))
folder = "./"

def only_video(folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print (device)
    
    data_transform = {
    'vid_train': transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
        transforms.RandomRotation(30),  # Randomly rotate up to 30 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'vid_val': transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'vid_test': transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    }

    datasets = ['vid_train','vid_val','vid_test']
    image_data = {}
    for x in datasets:
        image_data[x] = tv.datasets.ImageFolder(folder + '/' + x, transform=data_transform[x])


    data_loader = {}
    for x in datasets:
        data_loader[x] = torch.utils.data.DataLoader(image_data[x], batch_size=32,
                   shuffle=True, num_workers=0)

    
    net = Net()
    print(net)
    net.print_model_layers()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.0001) 
    
    #train(data_loader, criterion, net, device, optimizer)
    #test(data_loader, criterion, net, device, optimizer)

only_video(folder)
