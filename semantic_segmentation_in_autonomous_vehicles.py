"""

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
"""
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
