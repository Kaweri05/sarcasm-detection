import streamlit as st
import json
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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
st.write("Training model... please wait")
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

    # Language Detection
    text_lower = user_input.lower()

    hindi_words = [
        "kya", "hai", "fir",
        "waah", "wah", "acha",
        "kyu", "nahi"
    ]

    marathi_words = [
        "kay", "kaay", "kasa",
        "aahe", "ahe",
        "khup", "mast",
        "punha", "bara",
        "chan", "nahi",
        "zala", "zhali",
        "mhanje", "mala",
        "tula", "apla",
        "marathi"
    ]

    hindi_count = sum(
        word in text_lower
        for word in hindi_words
    )

    marathi_count = sum(
        word in text_lower
        for word in marathi_words
    )

    if marathi_count > hindi_count and marathi_count > 0:
        language = "Marathi"

    elif hindi_count > 0:
        language = "Hindi / Hinglish"

    else:
        language = "English"


    # Sentiment
    from textblob import TextBlob

    polarity = TextBlob(
        user_input
    ).sentiment.polarity

    if polarity > 0:
        sentiment = "😊 Positive"

    elif polarity < 0:
        sentiment = "😔 Negative"

    else:
        sentiment = "😐 Neutral"


    # Sarcasm Prediction
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

    confidence = float(
        prediction[0][0]
    ) * 100


    # Show Results
    st.subheader("Prediction Result")

    st.write(
        f"🌍 Language: {language}"
    )

    st.write(
        f"💭 Sentiment: {sentiment}"
    )

    if prediction[0][0] > 0.5:
        st.error("😏 Sarcastic")
        st.write(
            f"🎯 Confidence: {confidence:.2f}%"
        )

    else:
        st.success("🙂 Not Sarcastic")
        st.write(
            f"🎯 Confidence: {100-confidence:.2f}%"
        )