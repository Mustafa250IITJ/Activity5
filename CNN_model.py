# modified CNN model with three configuration in experiment branch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import tensorboard

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.USPS(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.USPS(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# CNN Models
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, config['conv1_channels'], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(config['conv1_channels'], config['conv2_channels'], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc_input_size = config['conv2_channels'] * 4 * 4

        self.fc1 = nn.Linear(self.fc_input_size, config['fc1_units'])
        self.fc2 = nn.Linear(config['fc1_units'], 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# train the Models
def train(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0
    device = next(model.parameters()).device
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            writer.add_scalar('training_loss', running_loss / 100, epoch * len(train_loader) + batch_idx)
            running_loss = 0.0

def test(model, test_loader, criterion, epoch, writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    targets = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            preds.extend(pred.squeeze().tolist())
            targets.extend(target.tolist())
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if writer:
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('accuracy', accuracy, epoch)
    return preds, targets


# evaluate the Models
def evaluate_model(model, test_loader, epoch):
    criterion = nn.CrossEntropyLoss()
    preds, targets = test(model, test_loader, criterion, epoch, None)
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='weighted')
    recall = recall_score(targets, preds, average='weighted')
    confusion = confusion_matrix(targets, preds)
    return accuracy, precision, recall, confusion


# plot the precision-recall curve and loss function using TensorBoard
def visualize_results(model_name, accuracy, precision):
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Precision: {precision:.4f}")

def main():
    num_epochs = 10
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # different CNN configurations
    cnn_configs = [
        {'conv1_channels': 32, 'conv2_channels': 64, 'fc1_units': 128},
        {'conv1_channels': 64, 'conv2_channels': 128, 'fc1_units': 256},
        {'conv1_channels': 64, 'conv2_channels': 128, 'fc1_units': 64}
    ]

    for i, config in enumerate(cnn_configs):
        model = CNN(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        writer = SummaryWriter(f'logs/cnn_{i}')

        for epoch in range(num_epochs):
            train(model, train_loader, nn.CrossEntropyLoss(), optimizer, epoch, writer)

            accuracy, precision, recall, confusion = evaluate_model(model, test_loader, epoch)

            visualize_results(f"CNN_{i}", accuracy, precision)

            print("CNN Confusion Matrix:")
            print(confusion)

        writer.close()


if __name__ == "__main__":
    main()


# tensorboard --logdir=logs
# tensorboard --logdir=logs
%load_ext tensorboard
%tensorboard --logdir logs
