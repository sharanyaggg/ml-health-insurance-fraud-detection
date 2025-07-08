import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load and preprocess the data
df = pd.read_csv('../data/sample_claims.csv')

# Encode categorical variables
le = LabelEncoder()
df['symptoms'] = le.fit_transform(df['symptoms'])
df['hospitalized'] = df['hospitalized'].map({'yes': 1, 'no': 0})

X = df[['age', 'symptoms', 'cost_estimation', 'hospitalized']]
y = df['is_fraudulent']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("Logistic Regression Results:")
print(classification_report(y_test, lr_pred))

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Decision Tree Results:")
print(classification_report(y_test, dt_pred))
