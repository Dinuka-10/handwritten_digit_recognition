import os
import struct
import random
import joblib
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# RANDOM SEED
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# DATASET CLASS
class MNISTDataset(Dataset):

    def __init__(self, image_file, label_file):

        with open(label_file, 'rb') as lb:
            magic, num = struct.unpack(">II", lb.read(8))
            self.labels = np.fromfile(lb, dtype=np.uint8)

        with open(image_file, 'rb') as img:
            magic, num, rows, cols = struct.unpack(">IIII", img.read(16))
            self.images = np.fromfile(img, dtype=np.uint8)
            self.images = self.images.reshape(num, rows, cols)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.images[idx] / 255.0

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return image, label

# LOAD DATASET
train_dataset = MNISTDataset(
    r"D:\dl_projects\handwritten_digit_recognition\model\datasets\train-images.idx3-ubyte",
    r"D:\dl_projects\handwritten_digit_recognition\model\datasets\train-labels.idx1-ubyte"
)

test_dataset = MNISTDataset(
    r"D:\dl_projects\handwritten_digit_recognition\model\datasets\t10k-images.idx3-ubyte",
    r"D:\dl_projects\handwritten_digit_recognition\model\datasets\t10k-labels.idx1-ubyte"
)

print("Training Samples:", len(train_dataset))
print("Testing Samples:", len(test_dataset))

# DATALOADER
g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    generator=g
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# MODEL
class DigitANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 10)

        )

    def forward(self, x):
        return self.network(x)

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# MODEL INIT
model = DigitANN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# TRAINING
epochs = 20

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}"
    )

# EVALUATION
model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        predictions.extend(
            preds.cpu().numpy()
        )

        actuals.extend(
            labels.numpy()
        )

accuracy = accuracy_score(
    actuals,
    predictions
)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")

print(
    classification_report(
        actuals,
        predictions
    )
)

# SAVE DIRECTORY
ARTIFACT_DIR = r"D:\dl_projects\handwritten_digit_recognition\model\artifacts"

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)

# SAVE MODEL (.pth)
torch.save(
    model.state_dict(),
    os.path.join(
        ARTIFACT_DIR,
        "digit_ann_model.pth"
    )
)

print("\nModel Saved!")

# CREATE & SAVE PIPELINE (.pkl) 
X_train = train_dataset.images.reshape(-1, 784)

pipeline = Pipeline([
    ("scaler", MinMaxScaler())
])

pipeline.fit(X_train)

joblib.dump(
    pipeline,
    os.path.join(
        ARTIFACT_DIR,
        "digit_pipeline.pkl"
    )
)

print("Pipeline Saved!")

# SAVE METADATA (.pkl)
metadata = {
    "model_name": "Handwritten Digit Recognition ANN",
    "input_features": 784,
    "hidden_layers": [512, 256, 128],
    "output_classes": 10,
    "epochs": epochs,
    "learning_rate": 0.001,
    "accuracy": round(accuracy * 100, 2)
}

joblib.dump(
    metadata,
    os.path.join(
        ARTIFACT_DIR,
        "model_metadata.pkl"
    )
)

print("Metadata Saved!")

print("\nAll Files Saved Successfully!")