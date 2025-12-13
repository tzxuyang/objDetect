import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch  # torch
import utils
from PIL import Image
import logging
from torchvision.transforms import ToPILImage
import os

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DinoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DinoClassifier, self).__init__()
        # Load the pre-trained DINOv3 model from timm
        self.backbone = timm.create_model('timm/vit_small_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)
        # Freeze the backbone parameters
        self.backbone.eval()  # set the model in evaluation mode
        self.num_classes = num_classes
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Define a simple classification head
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
    
    def get_transform(self):
        data_config = timm.data.resolve_model_data_config(self.backbone) 
        transforms = timm.data.create_transform(
            **data_config,
            is_training=True
        )
        return transforms
    
    def get_info(self):
        backbone_size = sum(param.numel() for param in self.backbone.parameters())
        head_size = sum(param.numel() for param in self.head.parameters())
        total_size = backbone_size + head_size
        return self.backbone.num_features, self.num_classes, total_size
    
    def process_image(self, file_path):
        data_config = timm.data.resolve_model_data_config(self.backbone) 
        transforms = timm.data.create_transform(
            **data_config,
            is_training=False
        )
        if isinstance(file_path, str):
            image = Image.open(file_path).convert('RGB')
        else:
            image = file_path.convert('RGB')
        input_tensor = transforms(image).unsqueeze(0).to(device)
        return input_tensor
    
    def predict(self, input_tensor, class_names=None):
        with torch.no_grad():
            output = self.forward(input_tensor)
            prob = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, 1)
        class_names = class_names
        # logging.info(f"Predicted: {class_names[predicted.item()]}, Confidence: {confidence.item():.4f}")
        return class_names[predicted.item()], confidence.item()
    
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        logging.info(f"Model saved to {file_path}")

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        # Initialize the dataset with data source
        self.data = data
        self.labels = labels # Use torch.long for classification labels

    @classmethod
    def fromDirectory(cls, data_dir, label_dir, transform=None):
        # Load data and labels from a directory
        file_list = utils.create_file_list(data_dir)
        data = []
        labels = []
        for file in file_list:
            # Load data and labels
            data.append(Image.open(file).convert('RGB'))
            label_file = file.replace(data_dir, label_dir).replace('.jpg', '.txt')
            with open(label_file, 'r') as f:
                label = int(f.read().strip())
                labels.append(label)
        if transform:
            data = [transform(img) for img in data]
        return cls(data, labels)

    def __len__(self):
        # Return the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve a single sample at the given index
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def train_model(custom_model, train_loader, **kwargs):
    custom_model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Adam to optimze the classification hear
    optimizer = optim.Adam(custom_model.head.parameters(), lr=kwargs.get('learning_rate', 0.001))
    num_epochs = kwargs.get('num_epochs', 150)
    for epoch in range(num_epochs):
        custom_model.train()  # set the model to train mode
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = custom_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    logging.info("Train completed")

if __name__ == "__main__":
    train_file_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/train"
    train_label_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/labels/train"

    # Load custom model
    custom_model = DinoClassifier(num_classes=6)
    dim, num_classes,size = custom_model.get_info()
    logging.info(f"Custom Dino Classifier model created. with vit dimension of {dim} and num_classes: {num_classes} and model size: {size/1e6}M parameters")
    logging.info(custom_model)
    

    # Prepare dataset and dataloader
    transforms = custom_model.get_transform()
    train_dataset = CustomDataset.fromDirectory(
        train_file_directory, 
        train_label_directory,
        transform= transforms
    )
    logging.info(f"train_dataset size : {int(train_dataset.__len__())}")
    sample, label = train_dataset.__getitem__(0)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=12
    )
    # Train the model
    train_config = {
        'learning_rate': 0.001,
        'num_epochs': 120
    }
    train_model(custom_model, train_loader, **train_config)

    # Save the trained model
    folders = os.listdir("./runs/cls")
    print(len(folders))
    if len(folders) > 1:
        folder_cnt = len(folders) -1
        print(folder_cnt)
        next_folder = f"train{folder_cnt}"
    else: #gitkeep
        next_folder = "train"
    os.makedirs(f"./runs/cls/{next_folder}/weights", exist_ok=True)
    path=f"./runs/cls/{next_folder}/weights/dino_classifier.pth"
    custom_model.save_model(path)
    logging.info(f"Model saved to {path}.")

    # Validate the model
    trained_model = DinoClassifier(num_classes=6)
    trained_model.load_state_dict(torch.load(path))
    trained_model.to(device)
    # trained_model.eval()

    logging.info("--------------------------------------------------------------------------------------------------")
    val_dir = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/val"
    image_file_list = utils.create_file_list(val_dir)
    for image_path in image_file_list:
        image = Image.open(image_path).convert('RGB')
        # image.show()
        input_tensor = trained_model.process_image(image_path)
        class_name, confidence = trained_model.predict(input_tensor, class_names=['unplugged', 'port1', 'port2', 'port3', 'port4', 'port5'])
        logging.info(f"{image_path} classified as {class_name} with confidence {confidence:.4f}")

    trained_model.eval()
    logging.info("--------------------------------------------------------------------------------------------------")
    val_dir = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/val"
    image_file_list = utils.create_file_list(val_dir)
    for image_path in image_file_list:
        image = Image.open(image_path).convert('RGB')
        # image.show()
        input_tensor = trained_model.process_image(image_path)
        class_name, confidence = trained_model.predict(input_tensor, class_names=['unplugged', 'port1', 'port2', 'port3', 'port4', 'port5'])
        logging.info(f"{image_path} classified as {class_name} with confidence {confidence:.4f}")