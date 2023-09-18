import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from dataloader import vizDataset


def load_dataset(split, transform):
    dataset = vizDataset(split=split, image_transform=transform)
    print(f"Data size: {len(dataset)} samples")
    return dataset


def get_data_loader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle
    )


def initialize_encoder(train_dataset, val_data):
    enc = OneHotEncoder(handle_unknown="ignore")
    train_dataset.change_y(
        enc.fit_transform(np.asarray(train_dataset.get_y()).reshape(-1, 1)).todense()
    )
    val_data.change_y(
        enc.transform(np.asarray(val_data.get_y()).reshape(-1, 1)).todense()
    )
    return enc


def display_samples(dataset, num_samples=5):
    for _ in range(num_samples):
        sample_questionID = random.randint(0, len(dataset) - 1)
        metadata = dataset.get_metadata(sample_questionID)
        filename = metadata["image"]
        image = Image.open(f"./{dataset.split}/{filename}").convert("RGB")

        plt.imshow(image)
        plt.axis(False)
        plt.show()

        question = metadata["question"]
        answer = dataset.get_processeddata(sample_questionID)[2]

        print(f"Q: {question}")
        print(f"A: {answer}")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    val_data = load_dataset(split="val", transform=transform)
    train_data = load_dataset(split="train", transform=transform)

    display_samples(val_data)
