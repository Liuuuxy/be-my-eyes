from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch
import matplotlib.pyplot as plt


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


def answer_question(image_path, question):
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    plt.axis(False)
    plt.show()
    print(f"Question: {question}")

    encoding = processor(image, question, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    values, idxs = torch.topk(logits, 3)

    for i in range(3):
        print(
            f"Predicted answer top {i + 1}: {model.config.id2label[idxs[0][i].item()]}"
        )


def compute_accuracy(val_data, k):
    correct = 0
    accuracy = 0

    for i in range(len(val_data)):
        sample_questionID = i
        filename = val_data.get_metadata(sample_questionID)["image"]
        image_path = f"./val/{filename}"
        question = val_data.get_metadata(sample_questionID)["question"]

        try:
            encoding = processor(image_path, question, return_tensors="pt")
            outputs = model(**encoding)
            logits = outputs.logits
            values, idxs = torch.topk(logits, k)

            for j in range(k):
                if (
                    val_data.get_processeddata(sample_questionID)[2]
                    == model.config.id2label[idxs[0][j].item()]
                ):
                    correct += 1
                    break
        except:
            pass

    print(f"top {k}: {100 * correct / len(val_data)}")


def vilt(val_data):
    # Answer a single question
    sample_questionID = 1
    filename = val_data.get_metadata(sample_questionID)["image"]
    image_path = f"./val/{filename}"
    question = val_data.get_metadata(sample_questionID)["question"]
    answer_question(image_path, question)

    # Compute accuracy for k=1 and k=3
    compute_accuracy(val_data, 1)
    compute_accuracy(val_data, 3)
