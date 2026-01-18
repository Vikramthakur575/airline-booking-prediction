import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("customer_booking.csv", encoding="latin1")

# -------------------------------
# FIX: Encode flight_day
# -------------------------------
day_mapping = {
    'Mon': 0,
    'Tue': 1,
    'Wed': 2,
    'Thu': 3,
    'Fri': 4,
    'Sat': 5,
    'Sun': 6
}
df['flight_day'] = df['flight_day'].map(day_mapping)

# -------------------------------
# Feature Engineering
# -------------------------------
df['is_weekend'] = df['flight_day'].isin([5, 6]).astype(int)

# Drop high-cardinality columns
df.drop(columns=['route', 'booking_origin'], inplace=True)

# Encode categorical columns
le_sales = LabelEncoder()
le_trip = LabelEncoder()

df['sales_channel'] = le_sales.fit_transform(df['sales_channel'])
df['trip_type'] = le_trip.fit_transform(df['trip_type'])

# -------------------------------
# Split Data
# -------------------------------
X = df.drop('booking_complete', axis=1)
y = df['booking_complete']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_prob = model.predict_proba(X_test)[:, 1]
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -------------------------------
# Save Model
# -------------------------------
with open("booking_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as booking_model.pkl")
