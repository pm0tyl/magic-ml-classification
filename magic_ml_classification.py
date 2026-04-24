import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input,Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

magic_gamma_telescope = fetch_ucirepo(id=159)

X = magic_gamma_telescope.data.features

y = magic_gamma_telescope.data.targets['class'] 

le = LabelEncoder()
y_encoded = le.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True,
    verbose=1)

history = model.fit(X_train, y_train, epochs=200,batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

plt.plot(history.history['loss'], label='loss (train)')
plt.plot(history.history['val_loss'], label='loss (test)')
plt.title('Błąd (loss)')
plt.xlabel('Epoka')
plt.ylabel('Wartość błędu')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy (train)')
plt.plot(history.history['val_accuracy'], label='accuracy (test)')
plt.title('Dokładność (accuracy)')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.tight_layout()
plt.show()

y_train_pred = np.round(model.predict(X_train))
y_test_pred = np.round(model.predict(X_test))

print("\n=== METRYKI DLA DANYCH TRENINGOWYCH ===")
print(f"Accuracy: ",accuracy_score(y_train, y_train_pred))
print(f"Precision: ",precision_score(y_train, y_train_pred))
print(f"Recall: ",recall_score(y_train, y_train_pred))
print("Confusion Matrix:\n",confusion_matrix(y_train, y_train_pred))

print("\n=== METRYKI DLA DANYCH TESTOWYCH ===")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
