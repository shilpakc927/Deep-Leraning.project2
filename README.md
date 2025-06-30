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

**Training Workflow**

1. **Tokenization**

    ```python
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(messages)
    sequences = tokenizer.texts_to_sequences(messages)
    ```

2. **Padding**

    ```python
    X = pad_sequences(sequences, maxlen=100)
    ```

3. **Model**

    ```python
    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    ```

4. **Training**

    ```python
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X, y, epochs=5, batch_size=32)
    ```

5. **Saving**

    ```python
    model.save("spam_model.h5")

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    ```

---
## 🌐 Flask Application (`app.py`)

The Flask app manages routing and predictions.

**Routes:**

- `/`  
  Displays `front.html`, the dashboard introduction.

- `/home`  
  Handles:  
  - **GET:** Shows prediction form in `home.html`.
  - **POST:** Receives input text, runs prediction, displays results.

**Example Prediction Flow:**
```python
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

sequence = tokenizer.texts_to_sequences([input_text])
padded = pad_sequences(sequence, maxlen=100)
prediction = model.predict(padded)
label = "Spam" if prediction > 0.5 else "Ham"
```
---

## 🖥️ HTML Templates

**`front.html`**

This is the introduction dashboard page.

- Title and welcome text
- Link/button to go to the prediction page (/home)

**`home.html`**

This page handles:

- Text input form
- Showing prediction results
- Thank you message after exit

**Behavior:**

- **When GET:** Shows the text input form
- **When POST:** Displays the prediction result and exit button.
- **When exit button clicked:** Shows a thank you message.

---
## ⚙️ How to Run
1. **Clone the repository:**
    ```python
    git clone https://github.com/your-username/spam_classifier.git
    cd spam_classifier/flask4

     ```

3. **Create a virtual environment:**
    ```python
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    ```
    
5. **Install dependencies:**
     ```python 
    pip install -r requirements.txt
      ```

7. **Train model (if needed):**
   - Open the Jupyter Notebook (spam RNN .ipynb).
   - Run all cells to generate spam_model.h5 and tokenizer.pkl.

8. **Run Flask app:**
     ```python
    python app.py
     ```
     
10. **Open your browser:**
    ```
    http://127.0.0.1:5000/
    ```
---
## 🧪 Workflow Summary
```
✅ Step 1: Open dashboard (/) – start detection button.
✅ Step 2: Go to input page (/home) – enter text and submit.
✅ Step 3: Prediction result displayed with label.
✅ Step 4: Click exit – shows thank you message.

```
---
## 🛠️ Technologies Used
- Python 3
- TensorFlow/Keras
- Flask
- HTML/CSS

 ---

## 📄 License
   MIT License
 
---

## 📁 Files Included
- `spam RNN.ipynb` – Jupyter Notebook for training
- `front.html` – html introduction page
- `home.html` -  input and prediction page
- `spam.csv` - dataset
- `spam_model.h5` - trained model
- `tokenizer.pkl' – tokenizer file
---
## 📸 Project Screenshots

- [Dashboard Page](Screenshot%201.png)
- [Upload Page](Screenshot%202.png)
- [Prediction Result](Screenshot%203.png)
- [Thank You Message](Screenshot%204.png)

---

## 📩 Contact

**Shilpa K C**  
[LinkedIn](https://www.linkedin.com/in/shilpa-kc) | [Email](shilpakcc@gmail.com)

For questions or suggestions, feel free to reach out.

✅ **How to use this:**
- Copy everything **inside the fences above** (including the triple backticks at the start and end).
- Save it as:
  README.md
- Place it in your project folder.
- Commit and push to GitHub.

✅ This is **one single README file** describing:
- Notebook
- Flask
- Two HTML pages
- Complete workflow

---
