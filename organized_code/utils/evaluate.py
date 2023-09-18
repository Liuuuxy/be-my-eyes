import torch
import numpy as np
import torch.nn as nn
import clip


def calculate_topk_accuracy_clip(k, val_data, clip_labels, preprocess, model, device):
    accurate = 0
    n_samples = 0

    for batch_id, (image, question, answer) in enumerate(val_data):
        image = preprocess(image).unsqueeze(0).to(device)
        text = clip.tokenize(clip_labels).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(k)

        if clip.tokenize(answer)[0][1].item() == indices.item():
            accurate += 1
        n_samples += 1

    return accurate / n_samples


def compute_top_k_accuracy(val_loader, vqa_model, enc, train_dataset, k=1):
    """
    Compute the top-k accuracy for a given model and validation data.

    Parameters:
    val_loader (DataLoader): The DataLoader for the validation set.
    vqa_model (nn.Module): The trained model.
    enc (OneHotEncoder): The trained OneHotEncoder for label transformation.
    train_dataset: The training dataset object.
    k (int): The top-k value for accuracy computation.

    Returns:
    float: The top-k accuracy.
    """
    predicted_total = []
    label_total = []

    with torch.no_grad():
        for j, (image_feature, text_feature, label) in enumerate(val_loader):
            features = torch.cat((image_feature, text_feature), 1).to("cuda")
            outputs = vqa_model(features)

            _, predicted = outputs.data.cpu().topk(k, dim=1)

            label = label.reshape([-1, train_dataset.get_y().shape[1]]).cpu()
            labels_again = torch.argmax(label, dim=1)

            predicted_total.append(outputs.data.cpu().numpy())
            label_total.append(label.numpy())

    predicted_total = np.concatenate(predicted_total)
    label_total = np.concatenate(label_total)

    predicted_total = enc.inverse_transform(predicted_total)
    label_total = enc.inverse_transform(label_total)

    accuracy = 0
    for i in range(len(val_data)):
        correct = 0
        for answer in val_data.get_metadata(i)["answers"]:
            if predicted_total[i] == answer["answer"]:
                correct += 1
        accuracy += min(1, correct / 3)

    accuracy = (accuracy / len(val_data)) * 100
    print(f"Top-{k} accuracy is: {accuracy}%")
    return accuracy


# Training Function
def train_model(
    train_loader, input_dim, output_dim, learning_rate=0.001, num_epochs=40
):
    """
    Train the VQA model.

    Returns:
    nn.Module: The trained model.
    """
    # Initialize model, loss function, and optimizer
    vqa_model = FeedforwardNeuralNetModel(input_dim, output_dim)
    vqa_model.to("cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vqa_model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for i, (image_feature, text_feature, label) in enumerate(train_loader):
            features = (
                torch.cat((image_feature, text_feature), 1).requires_grad_().to("cuda")
            )
            label = label.reshape([-1, output_dim]).to("cuda")

            optimizer.zero_grad()
            outputs = vqa_model(features)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch: {epoch}. Loss: {loss.item()}")

    return vqa_model


# Evaluation Function
def evaluate_model(model, val_loader, output_dim):
    """
    Evaluate the VQA model.

    Returns:
    float: The accuracy of the model on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for j, (image_feature, text_feature, label) in enumerate(val_loader):
            features = torch.cat((image_feature, text_feature), 1).to("cuda")
            outputs = model(features)

            _, predicted = outputs.data.cpu().topk(1, dim=1)
            labels_again = torch.argmax(label.reshape([-1, output_dim]).cpu(), dim=1)

            total += label.size(0)
            if torch.cuda.is_available():
                correct += (predicted.T[0].cpu() == labels_again).sum()
            else:
                correct += (predicted == label).sum()

        accuracy = 100 * correct.item() / total
        print(f"Validation Accuracy: {accuracy}%")

    return accuracy
