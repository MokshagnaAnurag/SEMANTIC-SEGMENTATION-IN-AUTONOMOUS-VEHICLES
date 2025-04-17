!pip install torch torchvision torchaudio segmentation-models-pytorch albumentations opencv-python matplotlib

!wget -c "http://data.apolloscape.auto/segmentation/dataset.zip" -O apollo.zip
!unzip apollo.zip -d /content/apolloscape

!wget -c "https://github.com/alexgkendall/SegNet-Tutorial/archive/master.zip" -O camvid.zip
!unzip camvid.zip -d /content/camvid

!wget -c "https://a2d2-data.s3.eu-central-1.amazonaws.com/camera_lidar_semantic.zip" -O a2d2.zip
!unzip a2d2.zip -d /content/a2d2

!wget -c "https://storage.googleapis.com/idd-dataset-releases/idd_semantic.zip" -O idd.zip
!unzip idd.zip -d /content/idd

!wget -c "http://synthia-dataset.net/download/synthia.zip" -O synthia.zip
!unzip synthia.zip -d /content/synthia

!pip install torch torchvision torchaudio segmentation-models-pytorch albumentations opencv-python matplotlib

import os
import torch
import torchvision.transforms as transforms
import albumentations as A
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from segmentation_models_pytorch import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to PyTorch tensor
        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

train_dataset = SemanticSegmentationDataset(
    image_dir="/content/camvid/SegNet-Tutorial-master/CamVid/train",
    mask_dir="/content/camvid/SegNet-Tutorial-master/CamVid/trainannot",
    transform=A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = SemanticSegmentationDataset(
    image_dir="/content/camvid/SegNet-Tutorial-master/CamVid/val",
    mask_dir="/content/camvid/SegNet-Tutorial-master/CamVid/valannot",
    transform=A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
)

model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, 19, kernel_size=1)  # 19 classes for segmentation
model = model.to(device)

model = Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=19)
model = model.to(device)

import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for images, masks in train_loader:
    print(images.shape, masks.shape)
    break  # Stop after the first batch

model.to(device)
print(next(model.parameters()).device)  # Check if model is on the correct device

masks = masks.long()  # Ensure masks are in correct format

masks = masks.float()

num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs} starting...")
    model.train()
    total_loss = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Processing batch {batch_idx+1}...")
        images, masks = images.to(device), masks.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, dict):  # Check if output is a dict
            outputs = outputs['out']

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

if torch.isnan(loss) or torch.isinf(loss):
    print("Error: Loss is NaN or Inf!")

num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs} starting...")
    model.train()
    total_loss = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Processing batch {batch_idx+1}...")

        images, masks = images.to(device), masks.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, dict):  # Check if output is a dict
            outputs = outputs['out']

        loss = criterion(outputs, masks)

        # Check for NaN or Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Error: Loss is NaN or Inf!")
            break

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)['out']

    predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    color_mask = cv2.applyColorMap((predicted_mask * 10).astype(np.uint8), cv2.COLORMAP_JET)

    combined = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

    cv2.imshow("Real-Time Segmentation", combined)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

torch.save(model.state_dict(), "/content/semantic_segmentation_model.pth")

model.load_state_dict(torch.load("/content/semantic_segmentation_model.pth"))

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
])

cap = cv2.VideoCapture("test_video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = process_frame(frame)  # Your model function
    cv2.imshow("Segmented Output", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

import torch
import numpy as np

def compute_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def compute_miou(pred, target, num_classes=19):  # Adjust num_classes based on dataset
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))  # Ignore if no pixels for this class
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)  # Ignore NaN values

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation  # or your dataset

# Define test dataset transformations
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  # Adjust based on your model input size
])

# Load the test dataset
test_dataset = VOCSegmentation(
    root="data",
    year="2012",
    image_set="val",  # Use 'train' for training data
    download=True,
    transform=test_transform,
    target_transform=test_transform
)

# Define DataLoader
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

model.eval()
total_acc, total_miou = 0, 0
num_samples = 0

with torch.no_grad():
    for images, masks in test_loader:  # Ensure you have a test_loader
        images, masks = images.to(device), masks.to(device).long()
        outputs = model(images)

        if isinstance(outputs, dict):
            outputs = outputs['out']

        preds = torch.argmax(outputs, dim=1)  # Convert logits to class indices

        total_acc += compute_accuracy(preds, masks)
        total_miou += compute_miou(preds, masks, num_classes=19)
        num_samples += 1

print(f"Test Accuracy: {total_acc / num_samples:.4f}")
print(f"Test mIoU: {total_miou / num_samples:.4f}")

import matplotlib.pyplot as plt

num_epochs = 10
epochs = range(1, num_epochs + 1)

# Replace with actual values from training logs
train_loss = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18, 0.15]
train_miou = [0.4, 0.45, 0.5, 0.55, 0.6, 0.63, 0.65, 0.7, 0.73, 0.75]

plt.figure(figsize=(10, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, marker='o', label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

# mIoU Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_miou, marker='o', color="green", label="mIoU")
plt.xlabel("Epochs")
plt.ylabel("mIoU")
plt.title("Mean IoU Curve")
plt.legend()

plt.show()

train_loss = []
train_miou = []

for epoch in range(num_epochs):
    model.train()
    total_loss, total_miou = 0, 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device).long()
        optimizer.zero_grad()

        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs['out']  # Extract output if in dictionary format

        if not isinstance(outputs, torch.Tensor):  # Debugging check
            raise TypeError(f"Unexpected output type: {type(outputs)}")

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_miou += compute_miou(preds, masks, num_classes=19)

    avg_loss = total_loss / len(train_loader)
    avg_miou = total_miou / len(train_loader)

    train_loss.append(avg_loss)
    train_miou.append(avg_miou)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, mIoU: {avg_miou:.4f}")

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F

def visualize_segmentation(model, dataloader, device, class_colors):
    model.eval()

    # Fetch a single batch of images and masks
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device).long()

    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs['out']

    preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Convert logits to class indices
    masks = masks.cpu().numpy()

    # Plot the images, ground truth, and predicted masks
    fig, axes = plt.subplots(len(images), 3, figsize=(12, len(images) * 4))

    for i in range(len(images)):
        img = F.to_pil_image(images[i].cpu())  # Convert tensor to PIL image

        pred_colored = class_colors[preds[i]]  # Apply color mapping
        gt_colored = class_colors[masks[i]]

        # Ensure the shape is (H, W, 3) for visualization
        pred_colored = np.array(pred_colored, dtype=np.uint8).squeeze()
        gt_colored = np.array(gt_colored, dtype=np.uint8).squeeze()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title("Predicted Segmentation")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

# Define random class colors for visualization
num_classes = 19  # Change based on dataset
class_colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)  # Random RGB colors

# Call the function
visualize_segmentation(model, test_loader, device, class_colors)
