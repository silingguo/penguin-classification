import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Page title
# -----------------------------

st.set_page_config(
    page_title="Penguin Species Classification",
    page_icon="🐧",
    layout="wide"
)

st.title("🐧📊 K-Nearest Neighbors Penguin Classifier")
st.write(
    "This app demonstrates how the K-nearest-neighbors algorithm "
    "can be used to classify penguin species."
)

# -----------------------------
# Load and prepare the data
# -----------------------------

@st.cache_data
def load_data():
    data_url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
    data = pd.read_csv(data_url)

    # Remove observations with missing values
    data = data.dropna()

    # Convert categorical variables to numerical variables
    data = pd.get_dummies(
        data,
        columns=["island", "sex"],
        drop_first=True,
        dtype=float
    )

    return data


df = load_data()

X = df.drop(columns=["species"])
y = df["species"]

# Split the data before standardization
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=101,
    stratify=y
)


# -----------------------------
# App tabs
# -----------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["Explore Data", "Choose K", "Evaluate Model", "Make a Prediction"]
)


# -----------------------------
# Tab 1: Explore the data
# -----------------------------

with tab1:
    st.header("Explore the Data")

    st.write("The first five observations are:")

    # Display original-style data
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Number of observations", df.shape[0])

    with col2:
        st.metric("Number of predictors", X.shape[1])

    with col3:
        st.metric("Number of classes", y.nunique())

    st.subheader("Class distribution")

    class_counts = y.value_counts().sort_index()

    st.bar_chart(class_counts)


    # -----------------------------
# Tab 2: Choose K
# -----------------------------

with tab2:
    st.header("Choose the Number of Neighbors")

    st.write(
        """
        The value of K determines how many nearby observations are used
        to classify a new observation. Cross-validation is used to compare
        different values of K using the training data.
        """
    )

    k_values = range(1, 31)
    validation_errors = []

    for k_value in k_values:
        validation_model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k_value))
        ])

        scores = cross_val_score(
            validation_model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        validation_errors.append(1 - scores.mean())

    best_k = list(k_values)[validation_errors.index(min(validation_errors))]

    st.success(
        f"The value of K with the smallest cross-validation error is {best_k}."
    )

    fig, ax = plt.subplots()

    ax.plot(
        list(k_values),
        validation_errors,
        marker="o",
        linestyle="--"
    )

    ax.set_xlabel("Number of neighbors, K")
    ax.set_ylabel("Cross-validation error rate")
    ax.set_title("Cross-validation Error Rate versus K")

    st.pyplot(fig)

    st.write(
        """
        Smaller values of K can produce a flexible model that is sensitive
        to individual observations. Larger values of K produce a smoother
        decision rule.
        """
    )


# -----------------------------
# Tab 3: Evaluate the model
# -----------------------------

with tab3:
    st.header("Evaluate the KNN Model")

    selected_k = st.slider(
        "Select a value of K",
        min_value=1,
        max_value=30,
        value=best_k
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=selected_k))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    error_rate = 1 - accuracy

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Test accuracy", f"{accuracy:.2%}")

    with col2:
        st.metric("Test error rate", f"{error_rate:.2%}")


    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(y_test, predictions)

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual: Adelie", "Actual: Chinstrap", "Actual: Gentoo"],
        columns=["Predicted: Adelie", "Predicted: Chinstrap", "Predicted: Gentoo"]
    )

    st.dataframe(matrix_df)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=["Adelie", "Chinstrap", "Gentoo"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3))



# -----------------------------
# Tab 4: Make a prediction
# -----------------------------

with tab4:
    st.header("Predict Penguin Species")

    st.write(
        """
        Enter the measurements and characteristics of a penguin
        to predict its species using the KNN model.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        bill_length = st.number_input(
            "Bill length (mm)",
            min_value=0.0,
            value=45.0
        )

        bill_depth = st.number_input(
            "Bill depth (mm)",
            min_value=0.0,
            value=17.0
        )

        flipper_length = st.number_input(
            "Flipper length (mm)",
            min_value=0.0,
            value=200.0
        )

        body_mass = st.number_input(
            "Body mass (g)",
            min_value=0.0,
            value=4000.0
        )

    with col2:
        island = st.selectbox(
            "Island",
            ["Biscoe", "Dream", "Torgersen"]
        )

        sex = st.selectbox(
            "Sex",
            ["female", "male"]
        )
        year = st.number_input(
            "Year",
            min_value=2007,
            max_value=2009,
            value=2008
        )

        

    if st.button("Predict Species"):
        new_penguin = pd.DataFrame({
            "bill_length_mm": [bill_length],
            "bill_depth_mm": [bill_depth],
            "flipper_length_mm": [flipper_length],
            "body_mass_g": [body_mass],
            "island_Dream": [1.0 if island == "Dream" else 0.0],
            "island_Torgersen": [1.0 if island == "Torgersen" else 0.0],
            "sex_male": [1.0 if sex == "male" else 0.0],
            "year":[year],
        })

        # Ensure the new observation has exactly the same columns as X
        new_penguin = new_penguin.reindex(
            columns=X.columns,
            fill_value=0
        )

        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=best_k))
        ])

        final_model.fit(X_train, y_train)

        prediction = final_model.predict(new_penguin)[0]
        probabilities = final_model.predict_proba(new_penguin)[0]

        st.success(
            f"The predicted penguin species is {prediction}."
        )

        probability_df = pd.DataFrame({
            "Species": ["Adelie", "Chinstrap", "Gentoo"],
            "Predicted probability": probabilities
        })

        st.dataframe(probability_df)
