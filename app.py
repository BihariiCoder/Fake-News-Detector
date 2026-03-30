import streamlit as st
from predict import predict_news

st.title("📰 Fake News Detector")

news = st.text_area("Enter News Headline")

if st.button("Check"):
    if news:
        result = predict_news(news)

        if result == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
    else:
        st.warning("Enter some text")