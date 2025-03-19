import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Veriyi yükleme ve işleme
df = pd.read_csv("BankNote_Authentication.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Veriyi standartlaştırma
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# İleri yayılma fonksiyonu
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def tanh(Z):
    return np.tanh(Z)

# Parametrelerin başlatılması
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(42)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters

# İleri yayılma fonksiyonu
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X.T) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# Kayıp fonksiyonu
def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = - (np.dot(np.log(A2), Y) + np.dot(np.log(1 - A2), (1 - Y))) / m
    cost = float(np.squeeze(cost))
    return cost

# Geri yayılma fonksiyonu
def backpropagation(X, Y, cache, parameters):
    m = X.shape[0]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2.T, A1.T) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dZ1 = np.dot(dZ2, W2) * (1 - np.power(A1, 2)).T
    dW1 = np.dot(dZ1.T, X) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    grads = {"dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}
    return grads

# Parametre güncelleme fonksiyonu
def update_parameters(parameters, grads, learning_rate=0.01):
    W1 = parameters["W1"] - learning_rate * grads["dW1"]
    b1 = parameters["b1"] - learning_rate * grads["db1"]
    W2 = parameters["W2"] - learning_rate * grads["dW2"]
    b2 = parameters["b2"] - learning_rate * grads["db2"]
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

# Modelin eğitilmesi
def nn_model(X, Y, n_x, n_h, n_y, n_steps=1000, print_cost=True):
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(n_steps):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backpropagation(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    
    return parameters

# Modeli test etme
parameters = nn_model(X_train, y_train, X_train.shape[1], n_h=6, n_y=1, n_steps=1000)

def predict(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    return (A2 > 0.5).astype(int)

# Tahmin ve değerlendirme
y_pred = predict(parameters, X_test)
print("Accuracy:", accuracy_score(y_test.flatten(), y_pred.flatten()))
print("Confusion Matrix:", confusion_matrix(y_test.flatten(), y_pred.flatten()))
print("Classification Report:", classification_report(y_test.flatten(), y_pred.flatten()))
