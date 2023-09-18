import torch, os, json
import matplotlib.pyplot as plt
import numpy as np

# from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor


class vizDataset2(torch.utils.data.Dataset):
    def __init__(self, split="val", image_transform=None, tokenizer=False):
        self.json_dir = "./" + split + ".json"
        self.image_dir = "./" + split + "/"
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.clipprocessor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            "cuda"
        )
        self.y = []

        # Category definitions of answer.
        self.categories = ["unanswerable", "answerable"]

        # Load JSON files.
        print("Loading %s ..." % self.json_dir, end="")
        self.jsondata = json.load(open(self.json_dir))
        print(" finished")
        self.metadata = []
        # process the json
        self.index = []
        for i in range(len(self.jsondata)):
            if self.jsondata[i]["answerable"]:
                image = self.jsondata[i]["image"]
                question = self.jsondata[i]["question"]
                answers = self.jsondata[i]["answers"]
                # only confident answers with one word which is not unanswerable
                confident_answer = [
                    a["answer"]
                    for a in answers
                    if a["answer_confidence"] == "yes"
                    and a["answer"] != "unanswerable"
                    and len(a["answer"].split()) == 1
                ]
                try:
                    answer = max(set(confident_answer), key=confident_answer.count)
                    self.metadata.append([image, question, answer, i])
                except:
                    continue

        # Pre-tokenizing all sentences.
        # See documentation for what encode_plus does and each of its parameters.
        print("Tokenizing...", end="")
        # self.tokenized_questions = list()
        # self.tokenized_answers = list()
        for i in range(len(self.metadata)):
            question = self.metadata[i][1]
            answer = self.metadata[i][2]
            self.y.append(answer)

        print(" finished")

    def __getitem__(self, index: int):
        # Load images on the fly.

        filename, question, answer, jsonind = self.metadata[index]
        img_path = self.image_dir + filename
        image = Image.open(img_path).convert("RGB")

        if self.tokenizer:
            images_feature = self.images_features[index]
            question_feature = self.question_features[index]
            return images_feature, question_feature, self.y[index]
        else:
            return image, self.metadata[index][1], self.metadata[index][2]

    def load_image_only(self, index: int):
        filename, question, answer, jsonind = self.metadata[index]
        img_path = self.image_dir + filename
        image = Image.open(img_path).convert("RGB")
        return image

    def get_processeddata(self, index: int):
        return self.metadata[index]

    def get_metadata(self, index: int):
        filename, question, answer, jsonind = self.metadata[index]
        return self.jsondata[jsonind]

    def get_y(self):
        return self.y

    def change_y(self, y):
        self.y = y
        return self.y

    def __len__(self):
        return len(self.metadata)
