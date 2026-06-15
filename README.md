
# 🤖 Multilingual Sarcasm Detection 

## 📌 Project Overview

This project is a **Multilingual Sarcasm Detection System** built using **Natural Language Processing (NLP)** and **Deep Learning (LSTM)**. It detects whether a sentence is **sarcastic or non-sarcastic**, identifies the **language**, performs **sentiment analysis**, and shows a **confidence percentage** for predictions.

Unlike traditional sarcasm detection systems that mainly support English, this project aims to support **multiple languages, especially Indian languages such as Hindi and Marathi**.

---

## 🚀 Features

✅ **Sarcasm Detection** (Sarcastic / Not Sarcastic)
✅ **Multilingual Support** (English, Hindi, Marathi, Hinglish)
✅ **Automatic Language Detection**
✅ **Sentiment Analysis** (Positive / Negative / Neutral)
✅ **Confidence Percentage Prediction**
✅ **Interactive Web Application using Streamlit**
✅ **Real-time Text Prediction**

---

## 🛠️ Tech Stack

| Category             | Technology                    |
| -------------------- | ----------------------------- |
| Programming Language | Python                        |
| Frontend/UI          | Streamlit                     |
| Deep Learning        | TensorFlow / Keras            |
| NLP                  | NLTK                          |
| Model                | LSTM (Long Short-Term Memory) |
| Sentiment Analysis   | TextBlob                      |
| Language Detection   | Langdetect                    |
| Data Processing      | NumPy, Pandas                 |
| Visualization        | Matplotlib, Seaborn           |
| Deployment           | Streamlit Cloud               |
| Version Control      | Git & GitHub                  |

---

## 📂 Dataset Used

This project uses the **Sarcasm Headlines Dataset** in JSON format.

Dataset contains:

* `headline`
* `is_sarcastic`

Example:

```json
{
  "headline": "Wow! What a great day",
  "is_sarcastic": 1
}
```

---

## ⚙️ How the Project Works

### Step 1: User Input

User enters a sentence in the Streamlit app.

### Step 2: Language Detection

The system automatically detects the language using **Langdetect**.

### Step 3: Text Preprocessing

Input text is cleaned and processed using NLP techniques:

* Lowercase conversion
* Tokenization
* Sequence conversion
* Padding

### Step 4: Sarcasm Prediction

The processed text is passed to the **LSTM model** for sarcasm prediction.

### Step 5: Sentiment Analysis

**TextBlob** analyzes whether the sentence is:

* 🙂 Positive
* 😞 Negative
* 😐 Neutral

### Step 6: Confidence Score

The system shows confidence percentage for prediction.

---

## 🧠 Model Architecture

```text
Input Text
      ↓
Text Preprocessing
      ↓
Tokenization
      ↓
Padding
      ↓
Embedding Layer
      ↓
LSTM Layer
      ↓
Dense Layer
      ↓
Prediction Output
```

---

## 📦 Required Libraries

Install dependencies using:

```bash
pip install -r requirements.txt
```

### requirements.txt

```txt
streamlit
tensorflow-cpu==2.15.0
numpy
pandas
scikit-learn
nltk
textblob
langdetect
protobuf==4.25.3
seaborn
matplotlib
```

---

## ▶️ Run the Project Locally

### Clone Repository

```bash
git clone https://github.com/Kaweri05/sarcasm-detection.git
```

### Go to Project Folder

```bash
cd sarcasm-detection
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
python -m streamlit run app.py
```

---

## 📸 Sample Output("Screenshot 2026-06-15 113307.png")

### Input:

```text
Waah! kya problem hai
```

### Output:

```text
🌍 Language: Hindi
🙂 Sentiment: Negative
🤖 Prediction: Sarcastic
📊 Confidence: 85%
```

---

## 🌍 Supported Languages

* English
* Hindi
* Marathi
* Hinglish (Mixed Language)

---

## 🎯 Real World Applications

* Social Media Monitoring
* Customer Review Analysis
* Smart Chatbots
* Brand Reputation Monitoring
* Sentiment & Emotion Analysis
* Mental Health Analysis

---

## ⚠️ Challenges Faced

* Multilingual sarcasm detection
* Marathi language prediction issues
* TensorFlow deployment compatibility
* Streamlit deployment setup

---

## 🔮 Future Enhancements

* Voice-based sarcasm detection
* Meme sarcasm detection
* Better Indian language support
* BERT / IndicBERT integration
* WhatsApp / Twitter integration

---

## 📁 Project Structure

```text
sarcasm-detection/
│── app.py
│── copy_of_sarcasm_detection.py
│── multilingual_sarcasm.py
│── test.py
│── Sarcasm.json
│── requirements.txt
│── runtime.txt
│── README.md
```

---

## 👨‍💻 Author

**Kaweri Harinkhede**
Computer Engineering Student
Passionate about **AI, NLP, and Machine Learning**

GitHub: https://github.com/Kaweri05

---

## ⭐ Conclusion

This project improves machine understanding of human communication by detecting **hidden sarcasm**, **language**, and **sentiment** using **NLP and Deep Learning** in a multilingual environment.
