import json
import numpy as np
from streamlit import text
from deep_translator import GoogleTranslator
from langdetect import detect

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split


# Load dataset
data = []

with open("sarcasm.json", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))


# Prepare data
headlines = []
labels = []

for item in data:
    headlines.append(item['headline'])
    labels.append(item['is_sarcastic'])


# Tokenization
vocab_size = 10000
max_length = 40

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(headlines)

sequences = tokenizer.texts_to_sequences(headlines)
padded = pad_sequences(sequences, maxlen=max_length)

labels = np.array(labels)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42
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

print("Training model...")

model.fit(X_train, y_train, epochs=5)

print("\nModel Ready ✅")


# Multilingual sarcasm prediction
def multilingual_predict(text):

    try:
        language = detect(text)
    except:
        language = "unknown"

    print("\nDetected Language:", language)

    # Translate to English
    translated_text = GoogleTranslator(
        source='auto',
        target='en'
    ).translate(text)

    print("Translated Text:", translated_text)

    # Prediction
    sequence = tokenizer.texts_to_sequences([translated_text])
    padded_text = pad_sequences(
        sequence,
        maxlen=max_length
    )

    prediction = model.predict(padded_text)

    if prediction[0][0] > 0.5:
        print("😏 Sarcastic")
    else:
        print("🙂 Not Sarcastic")


# Continuous input
while True:

    sentence = input("\nEnter sentence (type exit to quit): ")

    if sentence.lower() == "exit":
        break

    multilingual_predict(sentence)

    hindi_words = ["kya", "wah", "fir", "kyu"]
marathi_words = ["kay", "mast", "punha", "khup"]

text_lower = text.lower()

if any(word in text_lower for word in hindi_words):
    language = "Hindi/Hinglish"

elif any(word in text_lower for word in marathi_words):
    language = "Marathi"

else:
    language = "English"