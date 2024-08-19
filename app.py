import pandas as pd
import numpy as np
import streamlit as st
import chardet
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import nltk
from nltk.corpus import stopwords
import re
import torch
nltk.download('stopwords')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def preprocess_text(text):
    # Load Arabic stopwords
    stop_words = set(stopwords.words('arabic'))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Rejoin tokens into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# Load the AraBERT model
@st.cache_resource
def load_model():
    model_name = 'aubmindlab/bert-base-arabertv2'
    return SentenceTransformer(model_name)

# Function to encode sentences
def encode_sentences(model, sentences):
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings.cpu().tolist()

def main():
    st.set_page_config(page_title="Repeated Questions")

    st.title("🔎 Arabic Repeated Qs Detector")
    st.text("")
    st.text("")
    st.markdown("###### Upload Excel File and Get DataFrame with Repeated Questions")

    model = load_model()
    file = st.file_uploader("Upload here", type=["csv", "xlsx"])

    if file:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        df.columns = df.columns.str.strip()
        df_copy = df.copy()

        # Preprocess text and add cleaned column to the copy
        df_copy['cleaned_question'] = df_copy['السؤال'].apply(preprocess_text)

        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False

        col1, col2, col3 = st.columns([0.2, 5.6, 0.2])
        with col2:
            button_placeholder = st.empty()
            if not st.session_state.button_clicked:
                if button_placeholder.button("Categorize"):
                    st.text("")
                    with st.spinner("Processing..."):
                        st.session_state.button_clicked = True
                        button_placeholder.empty()

                        # Encode sentences from the cleaned column and compute similarity matrices
                        df_copy['embedding'] = encode_sentences(model, df_copy['cleaned_question'].tolist())
                        embeddings = np.array(df_copy['embedding'].tolist())
                        similarity_matrix = cosine_similarity(embeddings)
                        distance_matrix = 1 - similarity_matrix
                        distance_matrix = np.clip(distance_matrix, a_min=0, a_max=None)

                        # Perform clustering
                        threshold = 0.85
                        eps = 1 - threshold
                        dbscan = DBSCAN(eps=eps, min_samples=6, metric='precomputed')
                        clusters = dbscan.fit_predict(distance_matrix)

                        # Add clusters to the copy DataFrame
                        df_copy['category'] = clusters

                        # Handle noise points (outliers)
                        max_category = df_copy['category'][df_copy['category'] != -1].max() if len(
                            df_copy[df_copy['category'] != -1]) > 0 else 0
                        df_copy['category'] = df_copy['category'].apply(
                            lambda x: max_category + 1 if x == -1 else x + 1)

                        # Merge results back to the original DataFrame
                        df['category'] = df_copy['category']
                        df_sorted = df.sort_values(by='category')

                        st.dataframe(df_sorted)

                        # Save the sorted dataframe to an Excel file in memory
                        output = io.BytesIO()
                        df_sorted.to_excel(output, index=False, engine='xlsxwriter')
                        output.seek(0)

                        st.download_button(
                            label="Download Results",
                            data=output,
                            file_name="data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

                        st.session_state.button_clicked = False

if __name__ == "__main__":
    main()
