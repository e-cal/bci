# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%
CHANNELS = 8
SAMPLE_HZ = 250

# %%
class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes,
        channels=CHANNELS,
        sample_hz=SAMPLE_HZ,
        dropout_rate=0.5,
        kernel_len=SAMPLE_HZ // 2,
        tfilter1=8,
        n_spatial_filters=2,
        tfilter2=None,
        norm_rate=0.25,
        dropout_type="Dropout",
    ):
        super(EEGNet, self).__init__()
        self.nb_classes = n_classes
        self.channels = channels
        self.sample_hz = sample_hz
        self.dropout_rate = dropout_rate
        self.kernel_len = kernel_len
        self.tfilter1 = tfilter1
        self.n_spatial_filters = n_spatial_filters
        if tfilter2 is None:
            self.tfilter2 = tfilter1 * n_spatial_filters
        else:
            self.tfilter2 = tfilter2
        self.norm_rate = norm_rate

        if dropout_type == "Dropout2D":
            self.dropout = nn.Dropout2d
        elif dropout_type == "Dropout":
            self.dropout = nn.Dropout
        else:
            raise ValueError(
                "dropoutType must be one of Dropout2D or Dropout, passed as a string."
            )

        self.conv1 = nn.Conv2d(
            1,
            self.tfilter1,
            kernel_size=(1, self.kernel_len),
            padding=(0, self.kernel_len // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.tfilter1)
        self.depthwise = nn.Conv2d(
            self.tfilter1,
            self.tfilter1 * self.n_spatial_filters,
            kernel_size=(self.channels, 1),
            groups=self.tfilter1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.tfilter1 * self.n_spatial_filters)
        self.separable_conv = nn.Conv2d(
            self.tfilter1 * self.n_spatial_filters,
            self.tfilter2,
            kernel_size=(1, 16),
            padding=(0, 8),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.tfilter2)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(self.tfilter2 * 4, self.nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, kernel_size=(1, 4))
        x = self.dropout(p=self.dropout_rate)(x)

        x = self.separable_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, kernel_size=(1, 8))
        x = self.dropout(p=self.dropout_rate)(x)

        x = self.flatten(x)
        x = F.linear(
            x,
            self.dense.weight
            * torch.clamp(torch.norm(self.dense.weight), max=self.norm_rate)
            / torch.norm(self.dense.weight),
        )
        x = F.softmax(x, dim=1)

        return x


# %%
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# %%
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx]).long()
        return x, y


# %%
# Load CSV data
data_path = "../data/3-games.csv"
df = pd.read_csv(data_path)

# Keep only EEG columns for X and marker for y

X = df[[f"eeg{i}" for i in range(1, 9)]]
y = df["marker"]

# Split into train, validation and test sets
X_train = X.iloc[: int(len(df) * 0.6)]
y_train = y.iloc[: int(len(df) * 0.6)]

X_val = X.iloc[int(len(df) * 0.6) : int(len(df) * 0.8)]
y_val = y.iloc[int(len(df) * 0.6) : int(len(df) * 0.8)]

X_test = X.iloc[int(len(df) * 0.8) :]
y_test = y.iloc[int(len(df) * 0.8) :]

# %%
# Create PyTorch datasets and data loaders
train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)
test_dataset = EEGDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# Scale X using StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)

# %%
# Define the model
model = EEGNet(n_classes=2)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for batch_idx, (df, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(df)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * df.size(0)
        _, pred = torch.max(outputs, 1)
        train_acc += torch.sum(pred == target.data)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)

    # Validation
    val_loss = 0.0
    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (df, target) in enumerate(val_loader):
            outputs = model(df)
            loss = criterion(outputs, target)
            val_loss += loss.item() * df.size(0)
            _, pred = torch.max(outputs, 1)
            val_acc += torch.sum(pred == target.data)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_acc / len(val_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

# Testing
test_loss = 0.0
test_acc = 0.0
model.eval()
with torch.no_grad():
    for batch_idx, (df, target) in enumerate(test_loader):
        outputs = model(df)
        loss = criterion(outputs, target)
        test_loss += loss.item() * df.size(0)
        _, pred = torch.max(outputs, 1)
        test_acc += torch.sum(pred == target.data)

test_loss = test_loss / len(test_loader.dataset)
test_acc = test_acc / len(test_loader.dataset)

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
