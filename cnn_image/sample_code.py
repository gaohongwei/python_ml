import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)  
        self.relu = nn.ReLU()  
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  
        self.fc2 = nn.Linear(128, num_classes)  

    def forward(self, x):
        x = self.conv1(x)  
        x = self.relu(x)  
        x = self.pool(x)  
        x = self.conv2(x)  
        x = self.relu(x)  
        x = self.pool(x)  
        x = x.view(-1, 64 * 32 * 32)  
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x)  
        return x

    # Method to get the optimizer (Adam optimizer)
    def get_optimizer(self, learning_rate=0.001):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    # Method to get the loss function (CrossEntropyLoss for classification)
    def get_loss_fn(self):
        return nn.CrossEntropyLoss()

    # Method to save the model weights to a file
    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    # Method to load the model weights from a file
    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))

    # Method to freeze the convolutional layers (so that they don't get updated during training)
    def freeze_layers(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False

    # Method to unfreeze the convolutional layers (allow them to be updated again during training)
    def unfreeze_layers(self):
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.conv2.parameters():
            param.requires_grad = True

    # Method to set the device (e.g., GPU or CPU)
    def set_device(self, device):
        self.to(device)  # Move model to the specified device (e.g., 'cuda' or 'cpu')

    # Method to print the model summary
    def summary(self, input_size=(3, 32, 32)):
        from torchsummary import summary
        summary(self, input_size=input_size)

    # Method to perform a training step (forward pass, loss calculation, and backward pass)
    def train_step(self, batch, loss_fn, optimizer):
        inputs, labels = batch
        outputs = self(inputs)  # Forward pass
        loss = loss_fn(outputs, labels)  # Calculate loss
        
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update weights
        
        return loss.item()  # Return the loss for monitoring

    # Method to perform an evaluation step (no gradient calculation)
    def evaluate_step(self, batch, loss_fn):
        inputs, labels = batch
        outputs = self(inputs)  # Forward pass
        loss = loss_fn(outputs, labels)  # Calculate loss
        
        return loss.item(), outputs  # Return the loss and predictions
