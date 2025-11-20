# Command to run this Web app on a local host: ``streamlit run app2.py``

import streamlit as st
import pickle
from gensim import corpora
from gensim.utils import simple_preprocess

# --- Load models ---
sentiment_model = pickle.load(open("sentiment_model.pkl", "rb"))   
lda_model = pickle.load(open("lda_model.pkl", "rb"))               
dictionary = pickle.load(open("dictionary.pkl", "rb"))             

# --- UI ---
st.title("Restaurant Review Analyzer 🍽️")
st.write("Find out **what topic** your review belongs to and its **sentiment polarity**!")

user_input = st.text_area("Enter your restaurant review here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # ---- Sentiment Prediction ----
        sentiment_pred = sentiment_model.predict([user_input])[0]
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = label_map.get(sentiment_pred, sentiment_pred)

        # ---- Topic Detection ----
        tokens = simple_preprocess(user_input)
        bow_vector = dictionary.doc2bow(tokens)
        topic_probs = lda_model.get_document_topics(bow_vector)

        if topic_probs:
            topic_id, topic_prob = max(topic_probs, key=lambda x: x[1])
        else:
            topic_id, topic_prob = (None, 0)

        # Map topic IDs to names (update based on your LDA model order)
        topic_labels = {
            0: "Hospitality Experience",
            1: "Value for Money",
            2: "Food Taste",
            3: "Wait Time",
            4: "Cuisine Variety",
            5: "Customer Loyalty"
        }
        topic_name = topic_labels.get(topic_id, "Unknown Topic")

        # ---- Display Results ----
        st.subheader("💬 Sentiment:")
        st.write(f"**{sentiment}**")

        st.subheader("🧩 Topic:")
        st.write(f"**{topic_name}** ({topic_prob*100:.2f}% confidence)")
