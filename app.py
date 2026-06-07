import streamlit as st
import json
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# Load dataset
data = []

with open("Sarcasm.json", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))


# Prepare data
headlines = []
labels = []

for item in data:
    headlines.append(item['headline'])
    labels.append(item['is_sarcastic'])


# Tokenizer
vocab_size = 10000
max_length = 40

tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(headlines)

sequences = tokenizer.texts_to_sequences(headlines)

padded = pad_sequences(
    sequences,
    maxlen=max_length
)


# Build model
model = Sequential([
    Embedding(vocab_size, 64),
    LSTM(64),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train model
model.fit(
    padded,
    np.array(labels),
    epochs=5,
    verbose=0
)


# Streamlit UI
st.title("😏 Sarcasm Detection App")

st.write(
    "Enter a sentence to detect sarcasm"
)

user_input = st.text_input(
    "Enter text"
)

if st.button("Predict"):

    sequence = tokenizer.texts_to_sequences(
        [user_input]
    )

    padded_text = pad_sequences(
        sequence,
        maxlen=max_length
    )

    prediction = model.predict(
        padded_text
    )

    if prediction[0][0] > 0.5:
        st.error("😏 Sarcastic")
    else:
        st.success("🙂 Not Sarcastic")