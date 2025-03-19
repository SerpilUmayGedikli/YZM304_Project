import torch
import torch.nn as nn

class Model2LayerTanh(nn.Module):
    def __init__(self):
        super(Model2LayerTanh, self).__init__()
        # 2 katmanlı modelin tanımlanması
        self.hidden1 = nn.Linear(4, 8)  # İlk gizli katman, 4 giriş ve 8 nöron
        self.hidden2 = nn.Linear(8, 8)  # İkinci gizli katman, 8 nöron
        self.output = nn.Linear(8, 1)   # Çıkış katmanı, 1 nöron (binary sınıflandırma)
        self.activation = nn.Tanh()     # İlk iki katmanda tanh aktivasyonu

    def forward(self, x):
        x = self.activation(self.hidden1(x))  # İlk gizli katman + aktivasyon
        x = self.activation(self.hidden2(x))  # İkinci gizli katman + aktivasyon
        x = torch.sigmoid(self.output(x))     # Çıkış katmanı + sigmoid aktivasyonu
        return x
