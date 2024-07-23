# Amazon Review Sentiment Analysis (LSTM)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview

- This project is a sentiment analysis application that was built using amazon product review datasets. 
- It uses a Long Short-Term Memory (LSTM) neural network to predict the sentiment (positive or negative) of a given review.
- The application was built using TensorFlow and Streamlit, effectively making it a full stack deep learning project. 

## Features

- **Sentiment Prediction**: Classify the sentiment of a given review as positive or negative.
- **Translation**: Automatically detect the language of the review and translate it to English if necessary.
- **Confidence Score**: Display the model's prediction confidence score.
- **Prediction History**: View the history of predictions with the ability to expand and collapse the history section.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
    ```
    git clone https://github.com/mohammed-majid/LSTM_Sentiment_amazon.git
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the pre-trained model and tokenizer** and place them in the project directory:
    - `sentiment_analysis_model.h5`
    - `tokenizer.pkl`

5. **Run the Streamlit application**:
    ```
    streamlit run app.py
    ```
    **or**
    ```
    python3 -m streamlit run app.py
    ```

## Usage

1. **Open the Streamlit application** in your web browser.

2. **Enter a review text** in the provided text area.

3. **Click the "Predict Sentiment" button** to get the sentiment prediction and confidence score.

4. If the review is in a language other than English, the translated review will also be displayed.

5. **View the prediction history** by expanding the "View Prediction History" section.

## Acknowledgements

This project was developed using the following libraries and tools:
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Langdetect](https://pypi.org/project/langdetect/)
- [Googletrans](https://pypi.org/project/googletrans/)
- [Pickle](https://docs.python.org/3/library/pickle.html)

### Side Note
- Considering the size of the dataset used for this project, I was unable to commit it to this repository. In case you want to check it out, [Press here.](https://kaggle.com/datasets/arhamrumi/amazon-product-reviews)

