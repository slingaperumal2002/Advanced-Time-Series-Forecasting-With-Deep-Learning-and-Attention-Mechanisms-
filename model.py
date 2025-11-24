"""PyTorch Seq2Seq with Attention (simple implementation).
Encoder: LSTM
Decoder: LSTM with global attention (Luong-style)
Designed for multivariate forecasting where input and output are sequences of vectors.
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        outputs, (h, c) = self.lstm(x)
        return outputs, (h, c)

class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        score = torch.bmm(self.proj(encoder_outputs), decoder_hidden.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(score, dim=1)  # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(output_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.attn = LuongAttention(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, y_prev, hidden, encoder_outputs):
        # y_prev: (batch, 1, output_size)
        # hidden: (h, c) each (num_layers, batch, hidden_size)
        h = hidden[0][-1]  # (batch, hidden_size)
        context, attn_weights = self.attn(h, encoder_outputs)
        lstm_input = torch.cat([y_prev.squeeze(1), context], dim=1).unsqueeze(1)
        outputs, hidden = self.lstm(lstm_input, hidden)
        pred = self.out(outputs.squeeze(1))
        return pred, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=1, device='cpu'):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(output_size, hidden_size, num_layers)
        self.device = device
    def forward(self, src, target_seq_len, teacher_forcing_ratio=0.5):
        batch = src.size(0)
        encoder_outputs, hidden = self.encoder(src)
        # initialize first decoder input as last input step
        y_prev = src[:, -1:, :].clone()
        outputs = []
        attn_weights_all = []
        for t in range(target_seq_len):
            pred, hidden, attn_weights = self.decoder(y_prev, hidden, encoder_outputs)
            outputs.append(pred.unsqueeze(1))
            attn_weights_all.append(attn_weights.unsqueeze(1))
            y_prev = pred.unsqueeze(1)  # feed prediction
        outputs = torch.cat(outputs, dim=1)
        attn = torch.cat(attn_weights_all, dim=1)
        return outputs, attn
