import gradio as gr
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding

# Load dataset
df = pd.read_csv("train.csv")

X = df["comment_text"]
y = df[df.columns[2:]].values

# Text Vectorization
MAX_FEATURES = 200000

vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,
    output_mode="int"
)

vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

# Dataset pipeline
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.shuffle(160000).batch(16).prefetch(8)

train = dataset.take(int(len(dataset) * 0.8))

# Build Model
model = Sequential()
model.add(Embedding(MAX_FEATURES + 1, 32))
model.add(Bidirectional(LSTM(32, activation="tanh")))
model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(6, activation="sigmoid"))

model.compile(loss="BinaryCrossentropy", optimizer="Adam")

# Train model (1 epoch for demo)
model.fit(train, epochs=1)

# Prediction function
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    response = ""
    for idx, col in enumerate(df.columns[2:]):
        response += f"{col}: {results[0][idx] > 0.5}\n"

    return response

# Gradio Interface
interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(lines=2, placeholder="Enter comment"),
    outputs="text",
    title="Hate Speech Detection"
)

if __name__ == "__main__":
    interface.launch()
