import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from ..utils.evaluate import train_model, evaluate_model, compute_top_k_accuracy


# Classifier Model Definition
class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.lstm = nn.LSTM(input_size, 2048)
        self.dropout = nn.Dropout(0.15)
        self.fc1 = nn.Linear(2048, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


# Main Execution
def clip_lstm(train_loader, val_loader, enc, train_data, input_dim, output_dim):
    trained_model = train_model(train_loader, input_dim, output_dim)
    validation_accuracy = evaluate_model(trained_model, val_loader, output_dim)
