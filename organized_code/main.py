# Import necessary modules and functions
import importlib
import torchvision.transforms as transforms
from models.vilt_model import vilt
from models.clip_linear import clip_linear
from models.clip_lstm_model import clip_lstm
from utils.helper import get_data_loader, initialize_encoder
from utils.vizDataset2 import vizDataset2
from utils.evaluate import calculate_topk_accuracy_clip
from sklearn.preprocessing import OneHotEncoder
from utils.helper import load_dataset
import clip
import torch
import torch.optim as optim
import torch.nn as nn
import os


# Function to dynamically import a module and get the model initializing function
def get_model_initializer(module_name, function_name="initialize_model"):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


transform = transforms.Compose([transforms.ToTensor()])

val_data = load_dataset(split="val", transform=transform)
train_data = load_dataset(split="train", transform=transform)

# vilt
vilt(val_data)

# clip
batch_size = 256
learning_rate = 0.001
num_epochs = 40
enc = OneHotEncoder(handle_unknown="ignore")

importantflag = False
if not importantflag:
    enc = initialize_encoder(train_data, val_data)

train_loader = get_data_loader(train_data, batch_size, shuffle=True)
val_loader = get_data_loader(val_data, batch_size, shuffle=False)

input_dim = 1024
output_dim = train_data.get_y().shape[1]

# Train Motivated with CLIP-Linear
clip_linear(train_loader, val_loader, enc, train_data, input_dim, output_dim)

# Using CLIP+LSTM
clip_lstm(train_loader, val_loader, enc, train_data, input_dim, output_dim)

# Test with zero-shot CLIP
transform = transforms.Compose([transforms.ToTensor()])

val_data = vizDataset2(split="val", image_transform=transform, tokenizer=False)
clip_labels = [f"The answer to {idx[1]} is {idx[2]}." for idx in val_data]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

top1_accuracy = calculate_topk_accuracy_clip(
    1, val_data, clip_labels, preprocess, model, device
)
print(f"Top-1 Accuracy: {top1_accuracy:.4f}")

top3_accuracy = calculate_topk_accuracy_clip(
    3, val_data, clip_labels, preprocess, model, device
)
print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
