from collections import defaultdict
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
import spacy
from tqdm import tqdm
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "FacebookAI/xlm-roberta-base"
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=2, ignore_mismatched_sizes=True
)
model.to(device)


class LabeledTokenDataset(Dataset):
    def __init__(self, text_list, labels_list):
        self.text_list = text_list
        self.labels_list = labels_list
        if model_name == "bert-base-multilingual-cased":
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.unk_token = "[UNK]"
            self.pad_token = "[PAD]"
            self.mask_token = "[MASK]"
        else:
            self.sep_token = "</s>"
            self.cls_token = "<s>"
            self.unk_token = "<unk>"
            self.pad_token = "<pad>"
            self.mask_token = "<mask>"
        self.nlp = spacy.load("en_core_web_sm")
        self.len = len(text_list)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        text = self.text_list[index]
        labels = self.labels_list[index].copy()

        # tokenize each word one at a time and label each token accordingly

        doc = self.nlp(text)
        words = []
        for word in doc:
            if word.lemma_ == ",":
                continue
            words.append(word.lemma_)

        tokens = []
        token_labels = []
        for word, label in zip(words, labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_labels.extend([label] * len(word_tokens))

        token_labels = [False] + token_labels + [False]
        tokens = [self.cls_token] + tokens + [self.sep_token]

        if len(tokens) > 512:
            tokens = tokens[:512]
            token_labels = token_labels[:512]
        else:  # add appropriate padding padding
            tokens = tokens + [self.pad_token] * (512 - len(tokens))
            token_labels = token_labels + [False] * (512 - len(token_labels))

        mask = [1 if token != self.pad_token else 0 for token in tokens]

        return {
            "ids": torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).long(),
            "mask": torch.tensor(mask).long(),
            "labels": torch.tensor(token_labels).long(),
        }


def json_to_dict(path):
    jdata = json.load(open(path, "r"))
    data = defaultdict(list)
    for _, datum in jdata.items():
        data["original"].append(datum["original"])
        data["text_to_keep"].append(datum["original"])
        data["labels"].append(datum["labels"])

    return data


def get_correct(logits, target):
    predictions = F.sigmoid(logits).argmax(dim=2).cpu()
    return (predictions == target.cpu()).sum().item()


data = json_to_dict("../data-creation/quality_control_data.json")
labeled_text = list(zip(data["original"], data["labels"]))
random.shuffle(labeled_text)
N = len(labeled_text)

texts, labels = list(zip(*labeled_text))

trainset = LabeledTokenDataset(texts[: int(N * 0.9)], labels[: int(N * 0.9)])
valset = LabeledTokenDataset(texts[int(N * 0.9) :], labels[int(N * 0.9) :])

trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
valloader = DataLoader(valset, batch_size=4, shuffle=True)


optim = torch.optim.Adam(model.parameters(), lr=0.00001)

epochs = 100
max_acc = 0
for epoch in range(epochs):
    model.train()
    tr_loss = 0
    for input in tqdm(trainloader, leave=False):
        ids = input["ids"].to(device)
        masks = input["mask"].to(device)
        labels = input["labels"].to(device)

        outputs = model(input_ids=ids, attention_mask=masks, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        tr_loss += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

    correct = 0
    total = 0
    val_loss = 0
    model.eval()
    for input in tqdm(valloader, leave=False):
        with torch.no_grad():
            ids = input["ids"].to(device)
            masks = input["mask"].to(device)
            labels = input["labels"].to(device)

            outputs = model(input_ids=ids, attention_mask=masks, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            val_loss += loss.item()
            correct += get_correct(logits, labels)
            total += labels.shape[1] * labels.shape[0]

    val_acc = 100 * correct / total

    print(
        f"epoch {epoch} | tr_loss: {tr_loss / len(trainloader):.4f} | val_loss: {val_loss / len(valloader):.4f} | val_acc: {val_acc:.2f}%"
    )
    
    if val_acc > max_acc:
        print("new model saved")
        torch.save(model, open("model.pt", "wb"))
        max_acc = val_acc
    
