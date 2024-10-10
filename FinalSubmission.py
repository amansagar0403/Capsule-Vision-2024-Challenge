import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import os

# Check if a GPU is available for faster computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data paths for training and validation datasets
train_data_path = 'Dataset/training'
val_data_path = 'Dataset/validation'

# Define data augmentation and normalization for the training and validation datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
        transforms.RandomRotation(20),  # Randomly rotate images by ±20 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness, contrast, saturation, and hue
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Random affine transformations
        transforms.ToTensor(),  # Convert PIL images to tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet means and stds
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize validation images to 224x224 pixels
        transforms.CenterCrop(224),  # Center crop the validation images
        transforms.ToTensor(),  # Convert PIL images to tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet means and stds
    ]),
}

# Load the datasets using ImageFolder for both training and validation
train_dataset = datasets.ImageFolder(train_data_path, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_data_path, transform=data_transforms['val'])

# Create data loaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained models (ResNet and DenseNet)
resnet = models.resnet50(weights='DEFAULT')  
densenet = models.densenet121(weights='DEFAULT')  

# Modify the final layers of the models to extract features instead of class predictions
num_ftrs_resnet = resnet.fc.in_features
resnet.fc = nn.Identity()  # Remove the final classification layer

num_ftrs_densenet = densenet.classifier.in_features
densenet.classifier = nn.Identity()  # Remove the final classification layer

# Define a combined model class that integrates features from both ResNet and DenseNet
class CombinedModel(nn.Module):
    def __init__(self, resnet, densenet, num_classes):
        super(CombinedModel, self).__init__()
        self.resnet = resnet
        self.densenet = densenet
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout layer for regularization
            nn.Linear(num_ftrs_resnet + num_ftrs_densenet, 512),  # Fully connected layer combining both model outputs
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(0.5),  # Dropout layer for regularization
            nn.Linear(512, num_classes)  # Final output layer for classification
        )

    def forward(self, x):
        resnet_out = self.resnet(x)  # Get features from ResNet
        densenet_out = self.densenet(x)  # Get features from DenseNet
        combined_out = torch.cat((resnet_out, densenet_out), dim=1)  # Concatenate the outputs
        return self.fc(combined_out)  # Pass concatenated features through the fully connected layers

# Instantiate the combined model with the specified number of classes
combined_model = CombinedModel(resnet, densenet, len(train_dataset.classes)).to(device)

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(combined_model.parameters(), lr=0.0001, weight_decay=1e-4)  # Adam optimizer with weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)  # Learning rate scheduler

# Initialize lists to store training and validation history
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

# Set the number of epochs for training
num_epochs = 50

# Create a directory to save plots if it doesn't exist
if not os.path.exists('plotsFinal'):
    os.makedirs('plotsFinal')

# Train and validate the combined model
for epoch in range(num_epochs):
    combined_model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        
        optimizer.zero_grad()  # Clear previous gradients
        outputs = combined_model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        running_loss += loss.item()  # Accumulate loss
        _, preds = torch.max(outputs, 1)  # Get predictions
        correct += (preds == labels).sum().item()  # Count correct predictions
        total += labels.size(0)  # Count total examples
    
    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_loss_history.append(epoch_loss)  # Store training loss
    train_acc_history.append(epoch_acc)  # Store training accuracy

    # Validation phase
    combined_model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
            outputs = combined_model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            val_loss += loss.item()  # Accumulate validation loss
            _, preds = torch.max(outputs, 1)  # Get predictions
            val_correct += (preds == labels).sum().item()  # Count correct predictions
            val_total += labels.size(0)  # Count total examples
    
    # Calculate average validation loss and accuracy for the epoch
    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = val_correct / val_total
    val_loss_history.append(val_epoch_loss)  # Store validation loss
    val_acc_history.append(val_epoch_acc)  # Store validation accuracy

    # Print epoch metrics to track training progress
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, '
          f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.4f}')
    
    # Step the scheduler based on validation loss
    scheduler.step(val_epoch_loss)

# Plot training and validation accuracy and loss
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('plotsFinal/training_validation_loss.png')  # Save the loss plot

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_acc_history, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('plotsFinal/training_validation_accuracy.png')  # Save the accuracy plot
plt.show()

# Confusion matrix and ROC curve generation
combined_model.eval()  # Set model to evaluation mode
all_labels = []  # List to store all true labels
all_preds = []  # List to store all predictions
all_probs = []  # List to store all predicted probabilities
all_features = []  # List to store all features

with torch.no_grad():  # Disable gradient calculation for evaluation
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        outputs = combined_model(inputs)  # Forward pass to get outputs
        features = torch.cat((resnet(inputs), densenet(inputs)), dim=1)  # Get features from both models
        _, preds = torch.max(outputs, 1)  # Get predictions
        probs = nn.functional.softmax(outputs, dim=1)  # Calculate predicted probabilities
        all_labels.extend(labels.cpu().numpy())  # Store true labels
        all_preds.extend(preds.cpu().numpy())  # Store predictions
        all_probs.extend(probs.cpu().numpy())  # Store predicted probabilities
        all_features.append(features.cpu().numpy())  # Store features

# Convert features to a numpy array for further analysis
all_features = np.concatenate(all_features)

# t-SNE Visualization for feature embeddings
tsne = TSNE(n_components=2, random_state=42)  # Initialize t-SNE
features_tsne = tsne.fit_transform(all_features)  # Fit and transform the features

# Plotting t-SNE for visualizing feature embeddings
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=all_labels, cmap='jet', alpha=0.6)
plt.colorbar(scatter, ticks=range(len(train_dataset.classes)))  # Add color bar for classes
plt.title('t-SNE Visualization of Feature Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('plotsFinal/tsne_visualization.png')  # Save the t-SNE plot
plt.show()

# Generate confusion matrix to evaluate model performance
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('plotsFinal/confusion_matrix.png')  # Save the confusion matrix plot
plt.show()

# Print classification report to see detailed metrics
print('Classification Report:')
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# ROC Curve generation for multi-class classification
all_labels_bin = label_binarize(all_labels, classes=range(len(train_dataset.classes)))  # Binarize the labels for ROC
fpr = dict()  # Dictionary to store false positive rates
tpr = dict()  # Dictionary to store true positive rates
roc_auc = dict()  # Dictionary to store AUC values

# Calculate ROC curve and AUC for each class
for i in range(len(train_dataset.classes)):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], np.array(all_probs)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotting ROC Curve for multi-class classification
plt.figure(figsize=(10, 8))
for i in range(len(train_dataset.classes)):
    plt.plot(fpr[i], tpr[i], label=f'Class {train_dataset.classes[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line for chance level
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('plotsFinal/roc_curve.png')  # Save the ROC curve plot
plt.show()

# Save the trained model for future use
torch.save(combined_model.state_dict(), 'combined_model.pth')


# Testing phase
"""
# Load the test dataset using ImageFolder (ensure you have the correct path)
test_data_path = 'Dataset/test'  # Change to your test data path
test_dataset = datasets.ImageFolder(test_data_path, transform=data_transforms['val'])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the model on the test dataset
combined_model.eval()  # Set model to evaluation mode
test_loss = 0.0
test_correct = 0
test_total = 0
all_test_labels = []
all_test_preds = []
all_test_probs = []

with torch.no_grad():  # Disable gradient calculation for testing
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        outputs = combined_model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        test_loss += loss.item()  # Accumulate test loss
        
        _, preds = torch.max(outputs, 1)  # Get predictions
        test_correct += (preds == labels).sum().item()  # Count correct predictions
        test_total += labels.size(0)  # Count total examples
        
        probs = nn.functional.softmax(outputs, dim=1)  # Calculate predicted probabilities
        all_test_labels.extend(labels.cpu().numpy())  # Store true labels
        all_test_preds.extend(preds.cpu().numpy())  # Store predictions
        all_test_probs.extend(probs.cpu().numpy())  # Store predicted probabilities

# Calculate average test loss and accuracy
average_test_loss = test_loss / len(test_loader)
test_accuracy = test_correct / test_total

# Print test metrics
print(f'Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Generate confusion matrix for test dataset
conf_matrix_test = confusion_matrix(all_test_labels, all_test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Confusion Matrix')
plt.savefig('plotsFinal/test_confusion_matrix.png')  # Save the test confusion matrix plot
plt.show()

# Print classification report for test dataset
print('Test Classification Report:')
print(classification_report(all_test_labels, all_test_preds, target_names=test_dataset.classes))

# ROC Curve generation for test dataset
all_test_labels_bin = label_binarize(all_test_labels, classes=range(len(test_dataset.classes)))  # Binarize the labels for ROC
fpr_test = dict()  # Dictionary to store false positive rates
tpr_test = dict()  # Dictionary to store true positive rates
roc_auc_test = dict()  # Dictionary to store AUC values

# Calculate ROC curve and AUC for each class on test dataset
for i in range(len(test_dataset.classes)):
    fpr_test[i], tpr_test[i], _ = roc_curve(all_test_labels_bin[:, i], np.array(all_test_probs)[:, i])
    roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

# Plotting ROC Curve for test dataset
plt.figure(figsize=(10, 8))
for i in range(len(test_dataset.classes)):
    plt.plot(fpr_test[i], tpr_test[i], label=f'Test Class {test_dataset.classes[i]} (AUC = {roc_auc_test[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line for chance level
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('plotsFinal/test_roc_curve.png')  # Save the test ROC curve plot
plt.show()
"""