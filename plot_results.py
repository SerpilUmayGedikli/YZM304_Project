import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_history(histories, titles):
    plt.figure(figsize=(12, 5))
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f"{titles[i]} Train")
        plt.plot(history.history['val_accuracy'], label=f"{titles[i]} Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Model Karşılaştırması")
    plt.show()

# Eğitilmiş modellerin eğitim geçmişlerini yükle
model_2layer_tanh = tf.keras.models.load_model("models/model_2layer_tanh.h5")
model_3layer_tanh = tf.keras.models.load_model("models/model_3layer_tanh.h5")
model_2layer_relu = tf.keras.models.load_model("models/model_2layer_relu.h5")
model_3layer_relu = tf.keras.models.load_model("models/model_3layer_relu.h5")

# İşlenmiş veriyi yükle
data = np.load("data/processed_data.npz")
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# Modelleri yeniden eğitmeden, geçmişlerini alarak görselleştirme
history_2layer_tanh = model_2layer_tanh.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)
history_3layer_tanh = model_3layer_tanh.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)
history_2layer_relu = model_2layer_relu.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)
history_3layer_relu = model_3layer_relu.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)

histories = [history_2layer_tanh, history_3layer_tanh, history_2layer_relu, history_3layer_relu]
titles = ["2L Tanh", "3L Tanh", "2L ReLU", "3L ReLU"]

plot_history(histories, titles)
