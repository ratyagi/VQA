import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageCaptionDataset
from encoder import Encoder
from decoder import DecoderWithAttention
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import classification_report

def calculate_accuracy(predictions, ground_truths):
    """Calculate accuracy as the percentage of correct predictions."""
    correct = sum([1 for pred, gt in zip(predictions, ground_truths) if pred == gt])
    return correct / len(ground_truths)

def calculate_bleu(predictions, ground_truths):
    """Calculate BLEU scores for predicted and ground truth sentences."""
    bleu_scores = []
    chencherry = SmoothingFunction()
    for pred, gt in zip(predictions, ground_truths):
        bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=chencherry.method1)
        bleu_scores.append(bleu)
    return sum(bleu_scores) / len(bleu_scores)

def evaluate():
    """Evaluate the VQA model using accuracy, BLEU score, and classification metrics."""
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = ImageCaptionDataset(
        image_folder="data/Images",
        captions_file="data/captions_fixed.txt",
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=1)

    # Load models
    encoder = Encoder(embed_size=512).to("cuda")
    decoder = DecoderWithAttention(
        embed_size=512,
        vocab_size=len(dataset.word2idx),
        encoder_dim=2048,
        decoder_dim=512,
        attention_dim=256
    ).to("cuda")

    encoder.load_state_dict(torch.load("/content/drive/MyDrive/FinalProject/encoder.pth"))
    decoder.load_state_dict(torch.load("/content/drive/MyDrive/FinalProject/decoder.pth"))

    encoder.eval()
    decoder.eval()

    predictions = []
    ground_truths = []
    prediction_texts = []
    ground_truth_texts = []

    for images, captions in dataloader:
        images, captions = images.to("cuda"), captions.to("cuda")

        # Forward pass
        with torch.no_grad():
            features = encoder(images)
            outputs = decoder(features, captions)

        # Predicted answer
        pred_indices = outputs.argmax(dim=2).squeeze().tolist()
        gt_indices = captions.squeeze().tolist()

        pred_text = " ".join([dataset.idx2word[idx] for idx in pred_indices if idx in dataset.idx2word])
        gt_text = " ".join([dataset.idx2word[idx] for idx in gt_indices if idx in dataset.idx2word])

        predictions.append(pred_indices[0])  # First token as the predicted answer
        ground_truths.append(gt_indices[0])
        prediction_texts.append(pred_text)
        ground_truth_texts.append(gt_text)

    # Calculate metrics
    accuracy = calculate_accuracy(predictions, ground_truths)
    bleu_score = calculate_bleu(prediction_texts, ground_truth_texts)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"BLEU Score: {bleu_score:.4f}")

    # Classification Report
    labels = list(set(ground_truths + predictions))  # Unique labels in the data
    used_labels = [label for label in labels if label in ground_truths or label in predictions]
    print("\nClassification Report:")
    print(classification_report(
    ground_truths,
    predictions,
    labels=used_labels,
    target_names=[dataset.idx2word[idx] for idx in used_labels],
    zero_division=0
    ))

if __name__ == "__main__":
    evaluate()
