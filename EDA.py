import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset (FIXED ENCODING)
df = pd.read_csv("customer_booking.csv", encoding="latin1")

print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nTarget Distribution:\n", df['booking_complete'].value_counts(normalize=True))

# Target Distribution Plot
sns.countplot(x='booking_complete', data=df)
plt.title("Booking Completion Distribution")
plt.show()

# Impact of Add-on Services
services = [
    'wants_extra_baggage',
    'wants_preferred_seat',
    'wants_in_flight_meals'
]

for col in services:
    sns.barplot(x=col, y='booking_complete', data=df)
    plt.title(f"Booking Rate vs {col}")
    plt.ylabel("Booking Probability")
    plt.show()

# Length of Stay vs Booking
sns.boxplot(x='booking_complete', y='length_of_stay', data=df)
plt.title("Length of Stay vs Booking")
plt.show()

# Sales Channel vs Booking
sns.barplot(x='sales_channel', y='booking_complete', data=df)
plt.title("Booking Rate by Sales Channel")
plt.show()

# Correlation Heatmap (NUMERIC ONLY)
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 6))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False)
plt.title("Numeric Feature Correlation Heatmap")
plt.show()


print("\nEDA Completed Successfully ✔")
