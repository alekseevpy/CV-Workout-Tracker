import torch
import torch.nn as nn

class EnhancedPoseLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=256, num_layers=3, num_classes=23, dropout=0.4):
        super(EnhancedPoseLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Более глубокая LSTM архитектура
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Улучшенный attention механизм
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Более глубокая классифицирующая сеть
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]

        # Multi-head attention (улучшенная версия)
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1
        )

        # Взвешенная сумма с attention
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), lstm_out
        ).squeeze(1)

        # Классификация
        output = self.classifier(context_vector)
        return output