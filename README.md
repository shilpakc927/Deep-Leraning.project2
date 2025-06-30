# ✉️ Spam Message Classifier

This project is a complete end-to-end spam detection system built with:

- **Jupyter Notebook**: Training an RNN model to classify messages as spam or not spam.
- **Flask Web Application**: Serving the trained model for predictions.
- **HTML Frontend**: A 2-page interface for user interaction.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model-training)
- [Flask Application](#flask-application)
- [HTML Templates](#html-templates)
- [How To Run](#how-to-run)
- [Workflow Summary](#workflow-summary)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Files Included](#files-included)
- [Project Screenshots](#project-screenshots)

---

## 🎯 Project Overview

Spam Message Classifier provides an easy way to detect spam messages in real time.

**Workflow:**

1. **Model Training** (Jupyter Notebook):
   - Load and preprocess text dataset.
   - Train an RNN classifier.
   - Save the trained model (`spam_model.h5`) and tokenizer (`tokenizer.pkl`).

2. **Flask Application**:
   - Loads the saved model and tokenizer.
   - Provides a web interface to input text.
   - Returns prediction results.
   - Displays a thank you page after exiting.

---

## 🗂️ Project Structure
```
spam_classifier/
├── flask4/
│ ├── app.py # Flask backend
│ ├── spam_model.h5 # Trained model file
│ ├── tokenizer.pkl # Saved tokenizer for preprocessing
│ ├── spam.csv # Dataset file
│ └── templates/
│ ├── front.html # Landing page
│ └── home.html # Prediction and thank you page
├── spam RNN .ipynb # Jupyter Notebook for training
└── README.md # Project documentation
```

---

## 📊 Dataset

The project uses a **spam dataset** containing labeled text messages:

- Each row: message text and label (`spam` or `ham`).
- Labels are encoded for training.

You can use public datasets such as:
- [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🧠 Model Training (`spam RNN .ipynb`)

The RNN is implemented in Keras.

**Architecture Overview:**
- **Embedding Layer** for word embeddings.
- **LSTM Layer** to learn sequence patterns.
- **Dense Output Layer** with sigmoid activation.

**Training Workflow:**

1. **Tokenization:**
   ```python
   tokenizer = Tokenizer()
   tokenizer.fit_on_texts(messages)
   sequences = tokenizer.texts_to_sequences(messages)
   ```
2. **Padding:**
   ```python
      X = pad_sequences(sequences, maxlen=100)
   ```
3. **Model:**
   ```python
     model = Sequential()
     model.add(Embedding(vocab_size, 128))
     model.add(LSTM(64))
     model.add(Dense(1, activation='sigmoid'))
  ```
4. **Training:**
  ```python
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32)
  ```

5.**Saving:**
  ```python
    model.save("spam_model.h5")
    with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
 ```
