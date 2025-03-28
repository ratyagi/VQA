# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from collections import Counter
import csv

class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder='/content/drive/MyDrive/FinalProject/data/Images', captions_file='/content/drive/MyDrive/FinalProject/data/captions_fixed.txt', transform=None, subset_size=2000):
        self.image_folder = image_folder
        self.transform = transform
        self.captions = []
        self.image_ids = []

        with open(captions_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip blank or improperly formatted lines
                if not line or '\t' not in line:
                    continue
                img_id, caption = line.split('\t', 1)  # Use `split('\t', 1)` to avoid issues with captions containing tabs
                self.image_ids.append(img_id)
                self.captions.append(caption)

        if subset_size:
            self.image_ids = self.image_ids[:subset_size]
            self.captions = self.captions[:subset_size]

        self.word2idx, self.idx2word = self.build_vocab(self.captions)


    def build_vocab(self, captions):
        counter = Counter()
        for caption in captions:
            counter.update(caption.split())

        vocab = ["<pad>", "<start>", "<end>", "<unk>"] + [word for word, _ in counter.items()]
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        return word2idx, idx2word

    def encode_caption(self, caption):
        tokens = caption.split()
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in tokens]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        while True:
            img_path = os.path.join(self.image_folder, self.image_ids[idx])
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                    if self.transform:
                        image = self.transform(image)

                    caption = self.encode_caption(self.captions[idx])
                    return image, torch.tensor(caption, dtype=torch.long)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            else:
                print(f"Missing image: {img_path}")
            
            # Pick another random index to avoid failure
            idx = torch.randint(0, len(self.captions), (1,)).item()