# decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Transform encoder features
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Transform decoder hidden state
        self.full_att = nn.Linear(attention_dim, 1)  # Compute scalar alignment scores

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # Shape: (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # Shape: (batch_size, 1, attention_dim)
        att = self.full_att(torch.tanh(att1 + att2))  # Shape: (batch_size, num_pixels, 1)
        alpha = torch.softmax(att, dim=1)  # Shape: (batch_size, num_pixels, 1)
        context = (encoder_out * alpha).sum(dim=1)  # Shape: (batch_size, encoder_dim)
        return context, alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, vocab_size, encoder_dim=2048, decoder_dim=512, attention_dim=256):
        super(DecoderWithAttention, self).__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Initialize LSTM state
        h, c = torch.zeros(batch_size, self.lstm.hidden_size).to(encoder_out.device), \
               torch.zeros(batch_size, self.lstm.hidden_size).to(encoder_out.device)

        # Embedding for captions
        embeddings = self.embedding(captions)

        # Generate sequence
        outputs = torch.zeros(batch_size, captions.size(1), self.fc.out_features).to(encoder_out.device)
        for t in range(captions.size(1)):
            context, _ = self.attention(encoder_out, h)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            outputs[:, t, :] = self.fc(h)

        return outputs
