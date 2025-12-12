import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch  # torch
import utils
from PIL import Image
from torchvision.transforms import ToPILImage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = timm.create_model('timm/vit_small_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)
    
    input_temp = torch.randn((1, 3, 256, 256))
    print(model(input_temp).shape) # get the shape of the output
    
    # timm.create_model is used to
    model.eval()  # set the model in evaluation mode
    for param in model.parameters():
        param.requires_grad = False
    print("model loaded")
    print(sum(param.numel() for param in model.parameters()))
    return model

def def_model(model, num_classes):
    # Extract dinov3 features
    feature_dim = model.num_features
    print(f"feature dimension is {feature_dim}")
    # Set 10 classes
    num_classes = num_classes
    # Create a simple MLP
    classifier = nn.Sequential(
        nn.Linear(feature_dim, 256),
        # nn.ReLU(inplace=True), # inplace replace the input with output
        nn.ReLU(), # inplace replace the input with output
        nn.Dropout(p=0.1),
        nn.Linear(256, num_classes)
    )
    # Connect the backbone with the classifier head
    class CustomDINOv3(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            features = self.backbone(x)
            output = self.head(features)
            return output

    custom_model = CustomDINOv3(model, classifier)
    print("Defined classification model")
    return custom_model, num_classes, feature_dim

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

def train_model(custom_model, train_loader):
    custom_model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Adam to optimze the classification hear
    optimizer = optim.Adam(custom_model.head.parameters(), lr=0.001)
    num_epochs = 100
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    print("Train completed")

def process_image(data_config, file_path):
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

def classifier(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(prob, 1)
    class_names = ['unplugged', 'port1', 'port2', 'port3', 'port4', 'port5']
    print(f"Predicted: {class_names[predicted.item()]}")
    return class_names[predicted.item()]

if __name__ == "__main__":
    train_file_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/train"
    train_label_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/labels/train"
    model = load_model()
    custom_model, num_classes, feature_dim = def_model(model, 6)


    data_config = timm.data.resolve_model_data_config(model) 
    transforms = timm.data.create_transform(
        **data_config,
        is_training=True
    )

    train_dataset = CustomDataset.fromDirectory(
        train_file_directory, 
        train_label_directory,
        transform= transforms
    )
    print(train_dataset.__len__)
    sample, label = train_dataset.__getitem__(0)
    print(sample)
    print(label)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=12
    )

    train_model(custom_model, train_loader)

    print("-------------------------------------------------------------")
    val_dir = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/val"
    image_file_list = utils.create_file_list(val_dir)
    for image_path in image_file_list:
        image = Image.open(image_path).convert('RGB')
        image.show()
        input_tensor = process_image(data_config, image_path)
        class_name = classifier(custom_model, input_tensor)
        print(f"{image_path} classified as {class_name}")