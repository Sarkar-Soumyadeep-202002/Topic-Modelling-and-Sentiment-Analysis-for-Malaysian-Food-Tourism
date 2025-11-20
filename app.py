
import streamlit as st
import pickle

# Load the pipeline
model = pickle.load(open("sentiment_model.pkl", "rb"))

# --- UI ---
st.title("Sentiment Analysis App")
st.write("Type a review below to find out if it's Positive, Negative, or Neutral!")

# Text input
user_input = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Pipeline handles both vectorization and prediction
        prediction = model.predict([user_input])[0]

        # Optional label map (adjust if your model encodes differently)
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = label_map.get(prediction, prediction)

        # Display result
        st.subheader(f"💬 Sentiment: {sentiment}")
