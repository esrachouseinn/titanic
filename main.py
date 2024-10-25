import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Veri yükleme ve ilk hazırlıklar
df = pd.read_csv('train.csv')
df = df.drop(['Name', 'Ticket'], axis=1)

# EDA - Veri Seti İlk İnceleme
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Eksik değerleri doldurma
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Cabin harfini çıkarma ve One-Hot Encoding
df['Cabin_Letter'] = df['Cabin'].apply(lambda x: x[0])
df = pd.get_dummies(df, columns=['Cabin_Letter'], drop_first=True)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Label Encoding for Sex
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Age ve Fare sütunlarının dağılımını inceleme
plt.figure(figsize=(10,6))
sns.histplot(df['Age'], kde=True, bins=30, color='blue')
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['Fare'], kde=True, bins=30, color='green')
plt.title('Fare Distribution')
plt.show()

# Survived sınıf dağılımını görselleştirme
plt.figure(figsize=(6,6))
sns.countplot(x='Survived', data=df)
plt.title('Survived Count')
plt.show()

# Cinsiyet ve hayatta kalma oranları
plt.figure(figsize=(6,6))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survived by Gender')
plt.show()

# Hedef ve özellikleri ayırma
X = df.drop(['PassengerId', 'Survived', 'Cabin'], axis=1)
y = df['Survived']

# Standartlaştırma
scaler = StandardScaler()
X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

# Eğitim ve test setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest ile model eğitimi
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Doğruluk ve sınıflandırma raporu
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix görselleştirme
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# En iyi parametreler için RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

print('En İyi Parametreler:', random_search.best_params_)

# En iyi modeli kullanarak tahmin yapma
best_model = random_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Keras Modeli
keras_model = Sequential()
keras_model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
keras_model.add(BatchNormalization())
keras_model.add(Dropout(0.5))
keras_model.add(Dense(64, activation='relu'))
keras_model.add(Dense(1, activation='sigmoid'))

keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Keras model eğitimi
history = keras_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Keras Model Performans Görselleştirme
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# Age dağılımını görselleştirme (Survivors and Non-Survivors)
plt.figure(figsize=(10,6))
sns.histplot(df[df['Survived'] == 1]['Age'], bins=30, kde=False, color="blue", label="Survived")
sns.histplot(df[df['Survived'] == 0]['Age'], bins=30, kde=False, color="red", label="Did not survive")
plt.title('Age Distribution of Survivors and Non-Survivors')
plt.legend()
plt.show()
