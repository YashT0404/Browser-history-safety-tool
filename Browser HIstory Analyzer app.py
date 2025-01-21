import streamlit as st
import sqlite3
import os
import shutil
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
from urllib.parse import urlparse

# Function to process the entire browser history
def get_entire_browser_history(db_path):
    if not os.path.exists(db_path):
        st.error(f"Database file not found: {db_path}")
        return None

    # Create a temporary copy of the database
    temp_db_path = "temp_history.db"
    shutil.copy2(db_path, temp_db_path)

    # Connect to the SQLite database
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()

    # Query to extract URL, title, visit count, and last visit time
    query = """
    SELECT url, title, visit_count, last_visit_time
    FROM urls
    ORDER BY last_visit_time DESC;
    """
    cursor.execute(query)
    data = cursor.fetchall()

    # Convert timestamps and prepare DataFrame
    history = []
    for url, title, visit_count, last_visit_time in data:
        if last_visit_time:
            last_visit_time = datetime(1601, 1, 1) + timedelta(microseconds=last_visit_time)
        else:
            last_visit_time = None
        history.append((url, title, visit_count, last_visit_time))

    conn.close()
    os.remove(temp_db_path)  # Clean up the temporary file

    df = pd.DataFrame(history, columns=["URL", "Title", "Visit Count", "Last Visit Time"])
    return df

# Function to categorize risk levels
def categorize_risk(probs, threshold_low=0.3, threshold_high=0.7):
    categories = []
    for prob in probs:
        if prob < threshold_low:
            categories.append("Safe")
        elif prob < threshold_high:
            categories.append("Low Risk")
        else:
            categories.append("High Risk")
    return categories

# Streamlit app
st.title("Browser History and Phishing Detection App")

# Browser History Section
st.header("Analyze Your Entire Browser History")
db_file = st.file_uploader("Upload your Chrome History SQLite Database", type="db")
if db_file:
    temp_db_path = "uploaded_history.db"
    with open(temp_db_path, "wb") as f:
        f.write(db_file.getbuffer())
    history_df = get_entire_browser_history(temp_db_path)
    if history_df is not None:
        st.subheader("Your Visited Sites")
        st.write(history_df)

# Phishing Detection Section
st.header("Phishing Detection Model")

# Upload phishing dataset
uploaded_csv = st.file_uploader("Upload Phishing Dataset (CSV)", type="csv")
if uploaded_csv:
    phishing_data = pd.read_csv(uploaded_csv)

    # Display dataset information
    st.subheader("Dataset Overview")
    st.write(phishing_data.head())

    phishing_data = phishing_data.sample(n=60000, random_state=42)

    # Map labels to binary values
    phishing_data['Label'] = phishing_data['Label'].map({'bad': 1, 'good': 0})

    # Split the data into training and testing sets
    X = phishing_data['URL']
    y = phishing_data['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature extraction
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5), max_features=10000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_vect, y_train)

    # Evaluate model
    y_pred = rf_model.predict(X_test_vect)
    y_prob = rf_model.predict_proba(X_test_vect)[:, 1]  # Probabilities of being phishing
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy}")
    st.text("Classification Report:")
    st.text(report)

    # Save model and vectorizer
    model_path = 'phishing_rf_model.pkl'
    vectorizer_path = 'phishing_vectorizer.pkl'
    joblib.dump(rf_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    st.success(f"Model saved as {model_path}")
    st.success(f"Vectorizer saved as {vectorizer_path}")

    # Real-Time Prediction Section
    st.header("Real-Time URL Prediction")
    input_url = st.text_input("Enter a URL to predict its risk level:")
    if input_url:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            # Load the saved model and vectorizer
            loaded_model = joblib.load(model_path)
            loaded_vectorizer = joblib.load(vectorizer_path)

            # Transform the input URL
            input_url_vect = loaded_vectorizer.transform([input_url])

            # Predict risk level
            risk_prob = loaded_model.predict_proba(input_url_vect)[0, 1]
            risk_category = categorize_risk([risk_prob])[0]

            st.subheader("Prediction Result")
            st.write(f"**Risk Category:** {risk_category}")
            st.write(f"**Probability of being phishing:** {risk_prob:.2f}")
        else:
            st.error("Model or vectorizer file not found. Please train the model first.")

    # Classify URLs in the entire browser history as safe or unsafe
    if history_df is not None:
        st.subheader("Classify Entire Visited History")
        browser_urls = history_df['URL']

        # Transform URLs using the trained vectorizer
        browser_urls_vect = vectorizer.transform(browser_urls)

        # Predict probabilities and categorize risk
        browser_probs = rf_model.predict_proba(browser_urls_vect)[:, 1]
        history_df['Risk Category'] = categorize_risk(browser_probs)

        # Visualization: Pie chart of risk categories
        st.subheader("Risk Categorization")
        category_counts = history_df['Risk Category'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Risk Categories")
        st.pyplot(fig)

        # Highlight high-risk sites
        high_risk_sites = history_df[history_df['Risk Category'] == 'High Risk']
        if not high_risk_sites.empty:
            st.subheader("High-Risk Sites Detected")
            st.write(high_risk_sites[['URL', 'Title', 'Last Visit Time']])
        else:
            st.success("No high-risk sites detected!")