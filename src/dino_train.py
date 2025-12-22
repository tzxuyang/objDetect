import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch  # torch
import utils
from PIL import Image, ImageOps
import logging
from sklearn.metrics import classification_report, accuracy_score
import os
import wandb
import random
import numpy as np
from sklearn import linear_model
import pickle

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PROJECT_NAME = "dino_classifier_177_dino_small_new"
# _PROJECT_NAME = "dino_classifier_177_dino_large"
_WANDB_KEY = "93205eda06a813b688c0462d11f09886a0cf7ae8"
_EPOCH = 200
_LR_init = 0.0003
_LR_min = 0.0001
_LR_init = 0.0008
_LR_min = 0.0002
_BATCH_SIZE = 128
_BATCH_SIZE_VAL = 32
_NUM_WORKERS = 12
_SEED = 77

def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # set PYTHONHASHSEED env var
    torch.manual_seed(seed)
    # If using multiple GPUs, set all seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
    
    # For complete determinism (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(torch.rand(2))
    logging.info(f"Random seed set to {seed}")

class DinoClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim=1024):
        super(DinoClassifier, self).__init__()
        # Load the pre-trained DINOv3 model from timm
        self.backbone = timm.create_model('timm/vit_small_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)
        # self.backbone = timm.create_model('timm/vit_large_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)
        # Freeze the backbone parameters
        self.backbone.eval()  # set the model in evaluation mode
        self.num_classes = num_classes
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Define a simple classification head
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output, features
    
    def get_train_transform(self):
        data_config = timm.data.resolve_model_data_config(self.backbone) 
        transform = timm.data.create_transform(
            **data_config,
            is_training=False,
            # no_aug=False,
            # color_jitter=0.4,
        )
        return transform
    
    def get_val_transform(self):
        data_config = timm.data.resolve_model_data_config(self.backbone) 
        transform = timm.data.create_transform(
            **data_config,
            is_training=False
        )
        return transform
    
    def get_info(self):
        backbone_size = sum(param.numel() for param in self.backbone.parameters())
        head_size = sum(param.numel() for param in self.head.parameters())
        total_size = backbone_size + head_size
        return self.backbone.num_features, self.num_classes, total_size
    
    def process_image(self, data_config, file_path, new_size = None):
        transforms = timm.data.create_transform(
            **data_config,
            is_training=False
        )
        if isinstance(file_path, str):
            image = Image.open(file_path).convert('RGB')
        else:
            image = file_path.convert('RGB')
        if new_size is not None:
            origin_size = image.size
            ratio_min = min(new_size[0]/origin_size[0], new_size[1]/origin_size[1])
            resized_size = (int(origin_size[0]*ratio_min), int(origin_size[1]*ratio_min))
            padding_left = (new_size[0] - resized_size[0]) //2
            padding_top = (new_size[1] - resized_size[1]) //2
            padding_right = new_size[0] - resized_size[0] - padding_left
            padding_bottom = new_size[1] - resized_size[1] - padding_top
            image = image.resize(resized_size)
            image = ImageOps.expand(image, border=(padding_left, padding_top, padding_right, padding_bottom), fill='black')
        input_tensor = transforms(image).unsqueeze(0).to(device)
        return input_tensor
    
    def predict(self, input_tensor, return_feature = False, class_names=None):
        with torch.no_grad():
            output, features = self.forward(input_tensor)
            prob = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, 1)
        if return_feature:
            return class_names[predicted.item()], confidence.item(), features
        else:
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
    def fromDirectory(cls, data_dir, label_dir, transform=None, resize = None, scale = 1):
        # Load data and labels from a directory
        file_list = utils.create_file_list(data_dir)
        data = []
        labels = []
        for _ in range(scale):
            for file in file_list:
                # Load data and labels
                img = Image.open(file).convert('RGB')
                if resize is not None:
                    origin_size = img.size
                    ratio_min = min(resize[0]/origin_size[0], resize[1]/origin_size[1])
                    resized_size = (int(origin_size[0]*ratio_min), int(origin_size[1]*ratio_min))
                    padding_left = (resize[0] - resized_size[0]) //2
                    padding_top = (resize[1] - resized_size[1]) //2
                    padding_right = resize[0] - resized_size[0] - padding_left
                    padding_bottom = resize[1] - resized_size[1] - padding_top
                    img = img.resize(resized_size)
                    img = ImageOps.expand(img, border=(padding_left, padding_top, padding_right, padding_bottom), fill='black')
                data.append(img)
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

def init_wandb(project_name="dino_classifier", wandb_key=_WANDB_KEY, config=None):
    wandb.login(key=wandb_key, relogin=True)
    if config is not None:
        config_in=config,
    else:
        config_in={
            "architecture": "DINOv3 + Custom Head",
            "dataset": "Port Classification",
            "epochs": _EPOCH,
            "batch_size": _BATCH_SIZE,
            "learning_rate": _LR_init,
            "eta_min": _LR_min
        }
    wandb.init(
        project=project_name,
        config=config_in
    )

def train_model(custom_model, train_loader, val_loader, **kwargs):
    custom_model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Adam to optimze the classification hear
    optimizer = optim.Adam(custom_model.head.parameters(), lr=kwargs.get('learning_rate', 0.001))
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=kwargs.get('num_epochs', 150), eta_min=kwargs.get('eta_min', 0.0001))

    num_epochs = kwargs.get('num_epochs', 150)
    for epoch in range(num_epochs):
        custom_model.train()  # set the model to train mode
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = custom_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss_train = running_loss / len(train_loader)

        custom_model.eval() # set the model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = custom_model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        running_loss_val = running_loss / len(val_loader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {running_loss_train:.4f}, Val loss: {running_loss_val:.4f}")
        wandb.log({
            "train_loss": running_loss_train,
            "val_loss": running_loss_val,
            "lr": scheduler.get_last_lr()[0],
            "epoch": epoch
        })
        scheduler.step()
    logging.info("Train completed")

def test_model(custom_model, test_loader, class_names):
    label_list = [i for i in range(len(class_names))]
    custom_model.to(device)
    custom_model.eval()  # set the model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs, _ = custom_model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    report = classification_report(all_labels, all_preds, labels = label_list, target_names=class_names)
    accuracy = accuracy_score(all_labels, all_preds)
    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:\n" + report)

def extract_model_features(custom_model, test_loader):
    custom_model.to(device)
    custom_model.eval()  # set the model to evaluation mode
    feature_list = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            _, feature = custom_model(images)
            feature_list.append(feature.detach().cpu().numpy())
    return feature_list

def train_classifier(project_name, train_file_directory, train_label_directory, test_file_directory, test_label_directory, class_names, train_cluster =False, new_size = None, batch_size=_BATCH_SIZE, lr_max=_LR_init, lr_min=_LR_min, epoch=_EPOCH):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    set_seed(_SEED)
    logging.info(f"Training Dino Classifier model on device: {device} with seed {torch.seed()}")

    # init wandb
    init_wandb(project_name= project_name)

    # Load custom model
    num_classes = len(class_names)
    custom_model = DinoClassifier(num_classes=num_classes)
    dim, num_classes,size = custom_model.get_info()
    logging.info(f"Custom Dino Classifier model created. with vit dimension of {dim} and num_classes: {num_classes} and model size: {size/1e6}M parameters")
    logging.info(custom_model)

    # Prepare dataset and dataloader
    transforms = custom_model.get_train_transform()
    train_dataset = CustomDataset.fromDirectory(
        train_file_directory, 
        train_label_directory,
        transform = transforms,
        resize = new_size
    )
    transforms = custom_model.get_val_transform()
    val_dataset = CustomDataset.fromDirectory(
        test_file_directory, 
        test_label_directory,
        transform = transforms,
        resize = new_size
    )
    logging.info(f"train_dataset size : {int(train_dataset.__len__())}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=_NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=_BATCH_SIZE_VAL, 
        shuffle=True, 
        num_workers=_NUM_WORKERS
    )
    # Train the model
    train_config = {
        'learning_rate': lr_max,
        'num_epochs': epoch,
        'eta_min': lr_min
    }
    train_model(custom_model, train_loader, val_loader, **train_config)

    # Save the trained model
    folders = os.listdir("./runs/cls")
    logging.info(len(folders))
    if len(folders) > 1:
        folder_cnt = len(folders) -1
        logging.info(folder_cnt)
        next_folder = f"train{folder_cnt}"
    else: #gitkeep
        next_folder = "train"
    os.makedirs(f"./runs/cls/{next_folder}/weights", exist_ok=True)
    path=f"./runs/cls/{next_folder}/weights/dino_classifier.pth"
    custom_model.save_model(path)
    logging.info(f"Model saved to {path}.")

    # Validate the model
    trained_model = DinoClassifier(num_classes=num_classes)
    trained_model.load_state_dict(torch.load(path))
    trained_model.to(device)

    # inference on val dataset
    trained_model.eval()
    data_config = timm.data.resolve_model_data_config(trained_model.backbone)
    logging.info("--------------------------------------------------------------------------------------------------")
    val_dir = test_file_directory
    image_file_list = utils.create_file_list(val_dir)
    for image_path in image_file_list:
        input_tensor = trained_model.process_image(data_config, image_path, new_size=new_size).to(device)
        class_name, confidence = trained_model.predict(input_tensor, class_names=class_names)
        logging.info(f"{image_path} classified as {class_name} with confidence {confidence:.4f}")

    # evaluate on test dataset
    logging.info("*********************************************Train set report: *********************************************")
    transforms = custom_model.get_val_transform()
    train_dataset = CustomDataset.fromDirectory(
        train_file_directory, 
        train_label_directory,
        transform= transforms,
        resize = new_size
    )
    test_loader = DataLoader(
        train_dataset, 
        batch_size=len(train_dataset), 
        shuffle=False, 
        num_workers=_NUM_WORKERS
    )
    test_model(trained_model, test_loader, class_names=class_names)
    if train_cluster:
        feature_list = extract_model_features(trained_model, test_loader)
        features = np.array(feature_list)
        print(features)
        print(features.shape)
        print(features[0])

    # evaluate on test dataset
    logging.info("**********************************************Test set report: **********************************************")
    transforms = custom_model.get_val_transform()
    test_loader = DataLoader(
        val_dataset, 
        batch_size=len(val_dataset), 
        shuffle=False, 
        num_workers=_NUM_WORKERS
    )
    test_model(trained_model, test_loader, class_names=class_names)

    # Train Novelty detection model with SGDOneClassSVM
    if train_cluster:
        clf = linear_model.SGDOneClassSVM(random_state=_SEED, tol=None)
        clf.fit(features[0])
        path=f"./runs/cls/{next_folder}/weights/anormally_detect.pkl"
        with open(path, 'wb') as file:
            pickle.dump(clf, file)

if __name__ == "__main__":
    train_file_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/train"
    train_label_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/labels/train"
    test_file_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/val"
    test_label_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/labels/val"

    train_classifier(
        _PROJECT_NAME, 
        train_file_directory, 
        train_label_directory, 
        test_file_directory, 
        test_label_directory, 
        new_size=None,
        class_names=['unplugged', 'port1', 'port2', 'port3', 'port4', 'port5'],
        batch_size=_BATCH_SIZE, 
        lr_max=_LR_init, 
        lr_min=_LR_min, 
        epoch=_EPOCH
    )