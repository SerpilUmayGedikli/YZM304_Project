import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

# Veriyi yükle
data = load_iris()
X = data.data
y = (data.target == 0).astype(int)  # Sadece sınıf 0'ı pozitif kabul et

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi tensörlere dönüştürme
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Sinir ağı modelini tanımlama
class BankNoteNN(nn.Module):
    def __init__(self, input_size):
        super(BankNoteNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)  # İlk gizli katman
        self.fc2 = nn.Linear(10, 10)  # İkinci gizli katman
        self.fc3 = nn.Linear(10, 1)  # Çıkış katmanı
        self.sigmoid = nn.Sigmoid()  # Çıkış aktivasyonu

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # Tanh aktivasyonu
        x = torch.tanh(self.fc2(x))  # Tanh aktivasyonu
        x = self.sigmoid(self.fc3(x))  # Sigmoid aktivasyonu
        return x

# Model oluşturma
input_size = X_train.shape[1]
model = BankNoteNN(input_size)

# Kayıp fonksiyonu ve optimizasyon
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Modeli eğitme
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Modeli eğitim moduna al
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Eğitim sonrası test yapma (optional)
model.eval()  # Modeli test moduna al
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_test_pred = (y_test_pred > 0.5).float()
    accuracy = (y_test_pred.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
    print(f'Test Accuracy: {accuracy:.4f}')

# Modeli kaydet
torch.save(model.state_dict(), "models/bank_note_model.pth")
print("Model eğitim tamamlandı ve kaydedildi.")
