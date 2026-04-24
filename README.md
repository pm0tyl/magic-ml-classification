## Machine Learning – Gamma Telescope Classification

### Opis projektu
Projekt przedstawia zastosowanie modelu Machine Learning do klasyfikacji danych z rzeczywistego zbioru MAGIC Gamma Telescope (UCI Machine Learning Repository).  
Celem jest rozróżnienie zdarzeń gamma od tła (hadronów) na podstawie cech numerycznych.

---

### Zakres
- pobranie danych z repozytorium UCI
- przygotowanie danych (Label Encoding)
- skalowanie danych (StandardScaler)
- podział na zbiór treningowy i testowy (stratified split)
- budowa modelu sieci neuronowej (MLP)
- zastosowanie regularyzacji (Dropout)
- trenowanie modelu z EarlyStopping
- ewaluacja modelu:
  - accuracy
  - precision
  - recall
  - confusion matrix

---

### Technologie
- Python  
- TensorFlow / Keras  
- scikit-learn  
- NumPy  
- Matplotlib  

---

### Wyniki
Model osiąga wysoką skuteczność klasyfikacji na zbiorze testowym (accuracy ~88%).  
Zastosowanie skalowania danych oraz EarlyStopping poprawia stabilność treningu i ogranicza overfitting.

---

### Opis problemu
Zadanie klasyfikacji danych teleskopowych polega na odróżnieniu sygnałów pochodzących od promieniowania gamma od szumu tła.  
Jest to problem binarnej klasyfikacji często spotykany w analizie danych naukowych i systemach detekcji zdarzeń.

### Cel projektu
Rozwijanie umiejętności pracy z rzeczywistymi danymi, przygotowania danych oraz budowy i ewaluacji modeli Machine Learning.
