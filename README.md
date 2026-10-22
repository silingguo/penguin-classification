# KNN Penguin Species Classification App

This Streamlit application demonstrates the K-Nearest Neighbors (KNN) classification algorithm using the Palmer Penguins dataset.

We aim to classify three penguin species: Adelie, Chinstrap, and Gentoo. The original dataset contains 344 observations. After removing missing values, 333 observations remain. We then use 5-fold cross-validation to choose the best K and evaluate the model using the test data.

## Features

- Explore the dataset and class distribution
- Select K using cross-validation
- Evaluate the KNN classifier
- Display accuracy and confusion matrix
- Display a classification report
- Predict penguin species based on new measurements and characteristics


## Run locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```