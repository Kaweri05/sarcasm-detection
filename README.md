---
title: Sentify
emoji: 😏
colorFrom: yellow
colorTo: red
sdk: docker
app_file: server.py
pinned: false
---

<div align="center">

# 😏 Sentify
### Multilingual Sarcasm Detection Dashboard

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-black?logo=flask)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange?logo=tensorflow)](https://tensorflow.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/cleve05/Sentify)
[![GitHub](https://img.shields.io/badge/GitHub-Kaweri05-black?logo=github)](https://github.com/Kaweri05/sarcasm-detection)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com)

**Detect sarcasm, analyze sentiment, and translate text across 22 languages on any device.**

Live Demo: https://huggingface.co/spaces/cleve05/Sentify

GitHub: https://github.com/Kaweri05/sarcasm-detection

</div>

---

## Table of Contents

- [Overview](#overview)
- [Block Diagrams](#block-diagrams)
- [User Instructions](#user-instructions)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [Deployment](#deployment)
- [Supported Languages](#supported-languages)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Author](#author)

---

## Overview

Sentify is a full-stack AI web application that detects sarcasm in text using a deep learning LSTM model. It supports multiple languages including English, Hindi, Marathi, and Hinglish, with real-time translation across 22 languages, sentiment analysis, voice input, and an analytics dashboard accessible on both mobile and desktop.

---

## Block Diagrams

### 1. System Architecture

```
+----------------------------------------------------------+
|                     SENTIFY SYSTEM                       |
+------------------+------------------+--------------------+
|   FRONTEND       |    BACKEND       |    DATA LAYER      |
|   (Browser)      |    (Flask)       |                    |
|                  |                  |                    |
|  HTML/CSS/JS --> |  server.py       |  sarcasm.json      |
|  Plotly Charts   |  |-- /predict    |  (26,000 headlines)|
|  Web Speech API  |  |-- /history    |                    |
|  MyMemory API    |  +-- /           |  analysis_history  |
|                  |                  |  .json             |
|  Mobile UI       |  TensorFlow LSTM |                    |
|  Bottom Nav      |  TextBlob NLP    |  model_metrics     |
|  Responsive      |  Word-list Lang  |  .json             |
+------------------+------------------+--------------------+
```

---

### 2. ML Model Pipeline

```
  Raw Text Input
       |
       v
+--------------+
|  Tokenizer   |  vocab=10,000  oov_token=OOV
+--------------+
       |
       v
+--------------+
|   Padding    |  maxlen=40  post-padding
+--------------+
       |
       v
+-----------------+
| Embedding Layer |  10,000 vocab  64-dim vectors
+-----------------+
       |
       v
+-----------------+
|  LSTM 64 units  |  captures sequence context
+-----------------+
       |
       v
+-----------------+
| Dense ReLU 24   |  feature extraction
+-----------------+
       |
       v
+------------------+
| Dense Sigmoid 1  |  binary output 0 or 1
+------------------+
       |
       v
  Confidence Score 0 to 100 percent
```

---

### 3. Prediction Request Flow

```
  Browser                  Flask server.py            Storage
    |                            |                       |
    |-- POST /predict ---------->|                       |
    |   { text: ... }            |                       |
    |                            |-- detect_language()   |
    |                            |-- analyse_sentiment() |
    |                            |-- predict_sarcasm()   |
    |                            |                       |
    |                            |-- append_history() -->|
    |                            |              analysis_history.json
    |<-- JSON Response ----------|                       |
    |   language                 |                       |
    |   sentiment                |                       |
    |   sarcasm true or false    |                       |
    |   confidence 75.0          |                       |
    |   trendAlert false         |                       |
    |   plotData                 |                       |
    |                            |                       |
    |-- Render result and chart  |                       |
    |-- Trigger alert if 80 plus |                       |
```

---

### 4. Translation Flow

```
  User Types or Speaks Text
           |
           v
  +------------------+
  | Source Language  |  Auto-detect or manual
  | Selector         |  22 languages supported
  +------------------+
           |
           v
  +---------------------------+
  |  MyMemory Free API        |
  |  mymemory.translated.net  |
  |  GET q=text&langpair=     |
  |  src to tgt               |
  +---------------------------+
           |
           v
  +------------------+
  | Translated Text  |
  | shown in output  |
  +------------------+
           |
      +----+----+
      v         v
  Copy Text   Analyze for
              Sarcasm
                |
                v
          POST /predict
```

---

### 5. Language and Sentiment Analytics Flow

```
  analysis_history.json
              |
              v
  +-------------------------+
  |   Parse History Data    |
  +-------------------------+
             |
    +--------+--------+
    v        v        v
+--------+ +-------+ +---------+
|Language| |Senti- | | Sarcasm |
|Mix Page| |ment   | | Rate    |
|        | |Page   | | Stats   |
+--------+ +-------+ +---------+
    |         |           |
    v         v           v
 Donut     Pie Chart   Bar Chart
 Chart     Pos Neg Neu  by Language
 Bar Chart x Sarcasm
```

---

### 6. Alert System Flow

```
  Prediction Result
         |
         v
  +------------------------+
  |  sarcasm == true ?     |
  +------------------------+
         |
    YES  |   NO
     +---+    +-> No alert
     v
  +------------------------+
  |  confidence >= 80% ?   |  configurable in Settings
  +------------------------+
         |
    YES  |   NO
     +---+    +-> No alert
     v
  +------------------------+
  |  triggerAlert()        |
  |  Show red banner       |
  |  Increment badge       |
  |  Log to Alerts page    |
  +------------------------+
```

---

## User Instructions

```
                 +------------------+
                 |  Open Sentify   |
                 |  in Browser     |
                 +--------+--------+
                          |
                          v
                 +------------------+
                 |   Dashboard      |<--------------------------+
                 |   Home Page      |                          |
                 +--------+---------+                          |
                          |                                    |
         +----------------+---------------+                   |
         v                v               v                   |
  +------------+  +-------------+  +------------+            |
  | Type text  |  | Tap mic and |  | Tap        |            |
  | in box     |  | Speak       |  | Translator |            |
  +------+-----+  +------+------+  +------+-----+            |
         |                |               |                   |
         +--------+-------+               |                   |
                  v                       v                   |
          +-------------+        +----------------+           |
          | Click       |        | Select langs   |           |
          | Predict     |        | Click Translate|           |
          +------+------+        +-------+--------+           |
                 |                       |                    |
                 v                       v                    |
          +-------------+        +----------------+           |
          | View Result |        | View           |           |
          | Language    |        | Translation    |           |
          | Sentiment   |        | Click Analyze  |           |
          | Sarcastic   |        | for Sarcasm    |           |
          | Confidence  |        +-------+--------+           |
          +------+------+                |                    |
                 |                       +--------------------+
         +-------+-------+
         v               v
  +------------+  +-----------+
  |Confidence  |  | Click     |
  | above 80%  |  | Speak     |
  +-----+------+  | Result    |
        |         +-----------+
   YES  |   NO
    +---+   +--> Continue
    v
  +------------+
  | Alert      |
  | Banner     |
  | appears    |
  +-----+------+
        |
        v
  +----------+  +----------+  +----------+
  | Alerts   |  | History  |  | Language |
  | Page     |  | Search   |  | Sentiment|
  +----------+  +----------+  +----------+
```

---

## Features

| Feature | Description |
|---|---|
| Sarcasm Detection | LSTM model trained on 26,000 plus headlines |
| Multilingual | English Hindi Marathi Hinglish |
| Translator | 22 plus languages via MyMemory API |
| Voice Input | Speech-to-text via Web Speech API |
| Text-to-Speech | Reads results aloud |
| Analytics | Language mix sentiment confidence charts |
| Alerts | Auto-trigger on high-confidence sarcasm |
| Mobile First | Bottom nav hamburger menu responsive |
| Export CSV | Download full prediction history |
| Profile | Customizable user profile with badges |
| i18n UI | Interface in English Hindi Marathi Spanish French |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Backend | Flask 3.1.3 |
| ML Model | TensorFlow-CPU 2.15.0 Keras LSTM |
| NLP | NLTK TextBlob |
| Translation | MyMemory Free API |
| Frontend | HTML5 CSS3 Vanilla JavaScript |
| Charts | Plotly.js |
| Speech | Web Speech API browser-native |
| Dataset | Sarcasm Headlines Dataset Kaggle |
| Deployment | Hugging Face Spaces Docker |

---

## Project Structure

```
sarcasm-detection/
|-- server.py                Flask backend main entry point
|-- app.py                   Legacy Streamlit version
|-- requirements.txt         Python dependencies
|-- Dockerfile               Hugging Face deployment
|-- Procfile                 Render Railway deployment
|-- runtime.txt              Python 3.10 version pin
|-- sarcasm.json             Dataset 26000 plus headlines
|-- templates/
|   +-- index.html           Full dashboard UI
|-- static/
|   +-- style.css            Mobile-first styles
+-- backend/
    |-- model_metrics.json   Model accuracy
    +-- analysis_history.json Prediction history log
```

---

## Run Locally

```bash
git clone https://github.com/Kaweri05/sarcasm-detection.git
cd sarcasm-detection

pip install -r requirements.txt

mkdir -p backend
echo '{"accuracy": 78.34}' > backend/model_metrics.json
echo '[]' > backend/analysis_history.json

python server.py
```

Open browser at http://localhost:5000

---

## Deployment

| Platform | Status | Link |
|---|---|---|
| Hugging Face Spaces | Live | https://huggingface.co/spaces/cleve05/Sentify |
| GitHub | Source Code | https://github.com/Kaweri05/sarcasm-detection |
| Kaggle | Notebook | https://www.kaggle.com |

---

## Supported Languages

| Language | Detection | Translation | TTS |
|---|---|---|---|
| English | Yes | Yes | Yes |
| Hindi | Yes | Yes | Yes |
| Marathi | Yes | Yes | Yes |
| Spanish | No | Yes | Yes |
| French | No | Yes | Yes |
| Arabic | No | Yes | Yes |
| Japanese | No | Yes | Yes |
| 15 more | No | Yes | No |

---

## Future Enhancements

- BERT and IndicBERT for improved accuracy
- Voice-based sarcasm detection
- Meme sarcasm detection
- Twitter and WhatsApp integration
- Batch CSV analysis
- Full Indic language NLP support

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Author

Kaweri Harinkhede
Computer Engineering Student and AI NLP Enthusiast

GitHub: https://github.com/Kaweri05

Live Demo: https://huggingface.co/spaces/cleve05/Sentify

---

*Because sometimes Oh great does not mean great at all.* 😏
