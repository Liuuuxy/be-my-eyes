import torch, os, json
import matplotlib.pyplot as plt
import numpy as np

# from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor


class vizDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split="val",
        image_transform=None,
        tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32"),
    ):
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
        self.categories = ["unanswerable", "answerable"]
        print("Loading %s ..." % self.json_dir, end="")
        self.jsondata = json.load(open(self.json_dir))
        print(" finished")
        self.metadata = []
        self.index = []
        for i in range(len(self.jsondata)):
            if self.jsondata[i]["answerable"]:
                image = self.jsondata[i]["image"]
                question = self.jsondata[i]["question"]
                answers = self.jsondata[i]["answers"]
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
        print("Tokenizing...", end="")
        self.tokenized_questions = list()
        self.tokenized_answers = list()
        for i in range(len(self.metadata)):
            question = self.metadata[i][1]
            answer = self.metadata[i][2]
            self.y.append(answer)
            if self.tokenizer:
                """
                encoded_question = self.tokenizer.encode_plus(
                    question, add_special_tokens = True, truncation = True,
                    max_length = 256, padding = 'max_length',
                    return_attention_mask = True,
                    return_tensors = 'pt')
                encoded_answer = self.tokenizer.encode_plus(
                    answer, add_special_tokens = True, truncation = True,
                    max_length = 256, padding = 'max_length',
                    return_attention_mask = True,
                    return_tensors = 'pt')
                self.tokenized_questions.append(encoded_question)
                self.tokenized_answers.append(encoded_answer)
                """
        self.images_features = list()
        self.question_features = list()
        for i in range(0, len(self.metadata), 256):
            if i % 1024 == 0:
                print("processing:", i)
            image = []
            question = []
            for j in range(i, min(i + 256, len(self.metadata))):
                question.append(self.metadata[j][1])
                filename = self.metadata[j][0]
                img_path = self.image_dir + filename
                image.append(Image.open(img_path).convert("RGB"))
            question_tokens = self.clipprocessor(
                text=question, padding=True, images=None, return_tensors="pt"
            ).to("cuda")
            question_embadding = (
                self.clipmodel.get_text_features(**question_tokens)
                .detach()
                .cpu()
                .numpy()
            )
            image_preproccesed = self.clipprocessor(
                text=None, images=image, return_tensors="pt"
            )["pixel_values"].to("cuda")
            image_embadding = (
                self.clipmodel.get_image_features(image_preproccesed)
                .detach()
                .cpu()
                .numpy()
            )
            self.question_features.append(question_embadding)
            self.images_features.append(image_embadding)
        self.question_features = np.concatenate(self.question_features)
        self.images_features = np.concatenate(self.images_features)
        print(self.images_features.shape)
        print(" finished")

    def __getitem__(self, index: int):
        if self.tokenizer:
            images_feature = self.images_features[index]
            question_feature = self.question_features[index]
            # return image, questiontext, questiontext_mask, answertext, answertext_mask,images_feature,question_feature,self.y[index]
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
