# kaggle-project
import os
import csv
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths to datasets
train_data_paths = {
    "chimpanzee": "/kaggle/input/chimpanzee-image-dataset/Chimpanzee",
    "fox": "/kaggle/input/image-classification-64-classes-animal/image/fox",
    "gorilla": "/kaggle/input/image-classification-64-classes-animal/image/gorilla",
    "sheep": "/kaggle/input/image-classification-64-classes-animal/image/sheep",
    "horse": "/kaggle/input/animals10/raw-img/cavallo",
    "squirrel": "/kaggle/input/animals10/raw-img/scoiattolo",
    "rabbit": "/kaggle/input/cat-vs-rabbit/train-cat-rabbit/rabbit",
    "moose": "/kaggle/input/image-classification-dataset-32-classes/image/moose",
    "collie": "/kaggle/input/120-dog-breeds-breed-classification/Images/n02106030-collie",
    "antelope": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/antelope",
    "bat": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/bat",
    "beaver": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/beaver",
    "blue+whale": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/blue+whale",
    "bobcat": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/bobcat",
    "buffalo": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/buffalo",
    "chihuahua": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/chihuahua",
    "cow": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/cow",
    "dalmatian": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/dalmatian",
    "deer": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/deer",
    "dolphin": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/dolphin",
    "elephant": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/elephant",
    "german+shepherd": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/german+shepherd",
    "giant+panda": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/giant+panda",
    "giraffe": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/giraffe",
    "grizzly+bear": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/grizzly+bear",
    "hamster": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/hamster",
    "hippopotamus": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/hippopotamus",
    "humpback+whale": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/humpback+whale",
    "killer+whale": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/killer+whale",
    "leopard": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/leopard",
    "lion": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/lion",
    "mole": "/kaggle/input/image-classification-64-classes-animal/image/mole",
    "mouse": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/mouse",
    "otter": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/otter",
    "ox": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/ox",
    "persian+cat": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/persian+cat",
    "pig": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/pig",
    "polar+bear": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/polar+bear",
    "raccoon": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/raccoon",
    "rat": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/rat",
    "seal": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/seal",
    "siamese+cat": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/siamese+cat",
    "skunk": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/skunk",
    "spider+monkey": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/spider+monkey",
    "tiger": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/tiger",
    "walrus": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/walrus",
    "weasel": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/weasel",
    "wolf": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/wolf",
    "zebra": "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train/zebra",
 "rhinoceros": "/kaggle/input/image-classification-dataset-32-classes/image/rhinoceros"
}

limited_to_100 = list(train_data_paths.keys())  # All classes will be limited to 100 images

# Transformations for training and testing
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset to handle the 100 images limit
class CustomDataset(Dataset):
    def __init__(self, data_paths, transform=None, limit_classes=None, limit_count=170):
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = sorted(data_paths.keys())

        for label, class_name in enumerate(self.classes):
            path = data_paths[class_name]
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if limit_classes and class_name in limit_classes:
                files = files[:limit_count]
            self.data.extend(files)
            self.labels.extend([label] * len(files))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Create the dataset and data loader
train_dataset = CustomDataset(train_data_paths, transform=transform_train, limit_classes=limited_to_100)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

# Test dataset
class TestDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

test_dir = "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/test"
test_dataset = TestDataset(test_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Define the model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)  # Using ResNet50 here
num_classes = len(train_dataset.classes)  # Total classes based on the custom dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust the final fully connected layer
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training the model
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

train_model(model, train_loader, criterion, optimizer, num_epochs=30)

# Inference and CSV generation
def generate_csv(model, dataloader, output_csv):
    model.eval()
    prediction = []
    with torch.no_grad():
        for images, image_names in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            prediction.extend(zip(image_names, preds.cpu().numpy()))

    # Write to CSV
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "predicted_class"])
        for i, (image_name, pred_class) in enumerate(predictions, start=1):
            writer.writerow([f"{i:05d}.jpg", train_dataset.classes[pred_class]])

output_csv = "/kaggle/working/prediction.csv"
generate_csv(model, test_loader, output_csv)
print(f"Prediction saved to {output_csv}")
