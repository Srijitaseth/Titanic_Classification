import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


csv_file_path = '/Users/srijitaseth/Documents/Titanic_Classification/Titanic-Dataset.csv'


data = pd.read_csv(csv_file_path)


print("First few rows of the dataset:")
print(data.head())

data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)


data['Age'].fillna(data['Age'].median(), inplace=True)  
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)  

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}) 
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  


X = data.drop('Survived', axis=1)  
y = data['Survived']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nModel Evaluation:")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")

feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance in Titanic Survival Prediction')
plt.show()

new_passenger = pd.DataFrame({
    'Pclass': [3],
    'Sex': [0],  
    'Age': [22],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked': [2]  
})

survival_prediction = model.predict(new_passenger)
print(f"\nSurvival Prediction for the new passenger: {'Survived' if survival_prediction[0] == 1 else 'Did not survive'}")
