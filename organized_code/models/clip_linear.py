import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from ..utils.evaluate import train_model, evaluate_model, compute_top_k_accuracy


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


def clip_linear(train_loader, val_loader, enc, train_data, input_dim, output_dim):
    trained_model = train_model(train_loader, input_dim, output_dim)
    validation_accuracy = evaluate_model(trained_model, val_loader, output_dim)
    top1_acc = compute_top_k_accuracy(val_loader, train_model, enc, train_data, 1)
    top3_acc = compute_top_k_accuracy(val_loader, train_model, enc, train_data, 3)

    # Save model
    torch.save(trained_model.state_dict(), "best_model_so_far.pth")
