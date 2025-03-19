import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Veri setini yükleyelim
df = pd.read_csv(r"C:\Users\srplg\Desktop\YZM304_Project\data\BankNote_Authentication.csv")
 

# İlk 5 satırı görelim
print(df.head())

# Bağımsız değişkenler (X) ve bağımlı değişken (Y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçekleyelim (standartlaştırma)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
