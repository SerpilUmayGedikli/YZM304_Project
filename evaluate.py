import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# Veriyi yükle
data = np.load("data/processed_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Modeli yükle
model = BankNoteNN(input_size=X_test.shape[1])
model.load_state_dict(torch.load("models/bank_note_model.pth"))
model.eval()

# Test verisiyle tahmin yap
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    y_pred_prob = model(X_test_tensor)
    y_pred = (y_pred_prob > 0.5).float()

# Ölçüm metrikleri
accuracy = accuracy_score(y_test, y_pred.numpy())
recall = recall_score(y_test, y_pred.numpy())
precision = precision_score(y_test, y_pred.numpy())
f1 = f1_score(y_test, y_pred.numpy())
conf_matrix = confusion_matrix(y_test, y_pred.numpy())

# Sonuçları yazdır
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Sonuçları kaydet
with open("results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
