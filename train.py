# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ImageCaptionDataset
from encoder import Encoder
from decoder import DecoderWithAttention
from torchvision import transforms
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ImageCaptionDataset
from encoder import Encoder
from decoder import DecoderWithAttention

def collate_fn(batch):
    """
    Custom collate function to handle variable-length captions.
    Pads captions in the batch to the maximum length.
    """
    images, captions = zip(*batch)

    # Stack images into a tensor
    images = torch.stack(images)

    # Pad captions to the maximum length in the batch
    captions = [torch.tensor(caption, dtype=torch.long) for caption in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions

def train():
    dataset = ImageCaptionDataset("data/Images", "data/captions_fixed.txt", transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]),
    subset_size=2000 )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn )
    
    encoder = Encoder(embed_size=512).to(device)
    decoder = DecoderWithAttention(embed_size=512, vocab_size=len(dataset.word2idx), encoder_dim=2048, decoder_dim=512, attention_dim=256,).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    for epoch in range(10):
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()

            features = encoder(images)
            outputs = decoder(features, captions)
            loss = criterion(outputs.view(-1, outputs.size(2)), captions.view(-1))
            loss.backward()
            optimizer.step()

    # Save the trained models
    torch.save(encoder.state_dict(), "/content/drive/MyDrive/FinalProject/encoder.pth")
    torch.save(decoder.state_dict(), "/content/drive/MyDrive/FinalProject/decoder.pth")

    print("Model checkpoints saved to data/encoder.pth and data/decoder.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()
