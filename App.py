import gradio as gr
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model

# Load dataset (used only for column labels)
df = pd.read_csv("train.csv")

# Text vectorization setup
MAX_FEATURES = 200000

vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,
    output_mode="int"
)

X = df["comment_text"]
vectorizer.adapt(X.values)

# Load trained model
model = load_model("toxicity.h5")

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    response = ""
    for idx, col in enumerate(df.columns[2:]):
        response += f"{col}: {results[0][idx] > 0.5}\n"

    return response

interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(lines=2, placeholder="Enter comment to analyze"),
    outputs="text",
    title="Hate Speech Detection",
    description="Deep Learning based Toxic Comment Classifier using TensorFlow"
)

if __name__ == "__main__":
    interface.launch()
