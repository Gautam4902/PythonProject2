import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

df = pd.read_csv("features/train_features.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32)
model.save("deepfake_model.h5")