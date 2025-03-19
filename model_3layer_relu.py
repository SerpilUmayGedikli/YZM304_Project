import torch
import torch.nn as nn

class Model3LayerReLU(nn.Module):
    def __init__(self):
        super(Model3LayerReLU, self).__init__()
        # 3 katmanlı modelin tanımlanması
        self.hidden1 = nn.Linear(4, 8)  # İlk gizli katman, 4 giriş ve 8 nöron
        self.hidden2 = nn.Linear(8, 8)  # İkinci gizli katman, 8 nöron
        self.hidden3 = nn.Linear(8, 8)  # Üçüncü gizli katman, 8 nöron
        self.output = nn.Linear(8, 1)   # Çıkış katmanı, 1 nöron (binary sınıflandırma)
        self.activation = nn.ReLU()     # Gizli katmanlarda ReLU aktivasyonu

    def forward(self, x):
        x = self.activation(self.hidden1(x))  # İlk gizli katman + aktivasyon
        x = self.activation(self.hidden2(x))  # İkinci gizli katman + aktivasyon
        x = self.activation(self.hidden3(x))  # Üçüncü gizli katman + aktivasyon
        x = torch.sigmoid(self.output(x))     # Çıkış katmanı + sigmoid aktivasyonu
        return x
