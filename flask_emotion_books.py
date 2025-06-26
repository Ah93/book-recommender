from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import json
import math
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import os

# ───────────────────────────────────────────────
# ✅ Custom SelfAttention Layer (used in training)
# ───────────────────────────────────────────────
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1))
        alpha = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * alpha, axis=1)

# ───────────────────────────────────────────────
# ✅ Load Trained Model
# ───────────────────────────────────────────────
model = load_model(
    "emotion_model_1.h5",
    custom_objects={"SelfAttention": SelfAttention, "mse": MeanSquaredError()}
)

# ───────────────────────────────────────────────
# ✅ Load Saved Tokenizer (from training)
# ───────────────────────────────────────────────
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())

# ───────────────────────────────────────────────
# ✅ Load Books from Local JSON
# ───────────────────────────────────────────────
with open("books.json", "r", encoding="utf-8") as f:
    books_data = json.load(f)

# ───────────────────────────────────────────────
# ✅ Clean JSON for NaNs
# ───────────────────────────────────────────────
def clean_book(book):
    return {
        k: (v if not (isinstance(v, float) and math.isnan(v)) else None)
        for k, v in book.items()
    }

# ───────────────────────────────────────────────
# ✅ Flask App Setup
# ───────────────────────────────────────────────
app = Flask(__name__)
MAX_TOKENS = 128

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.json.get("emotion_text")
    if not user_input:
        return jsonify({"error": "No input provided."}), 400

    try:
        # Tokenize input
        sequences = tokenizer.texts_to_sequences([user_input])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=MAX_TOKENS, padding="post", truncating="post"
        )

        # Predict emotion vector
        pred_vector = model.predict(padded, verbose=0)[0]

        # Compute cosine similarities
        similarities = []
        for book in books_data:
            if "emotion_vector" not in book:
                continue
            book_vector = np.array(book["emotion_vector"])
            similarity = cosine_similarity([pred_vector], [book_vector])[0][0]
            similarities.append((similarity, clean_book(book)))

        top_books = sorted(similarities, key=lambda x: -x[0])[:5]
        recommended = [book for _, book in top_books]

        return jsonify({"books": recommended})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ───────────────────────────────────────────────
# ✅ Run Flask App
# ───────────────────────────────────────────────
import os

if __name__ == "__main__":
    # Grab the port from the environment (Render sets $PORT)
    port = int(os.environ.get("PORT", 5000))
    # Listen on 0.0.0.0 so it’s reachable externally
    app.run(host="0.0.0.0", port=port, debug=True)

