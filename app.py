import torch
import pandas as pd
import numpy as np
import streamlit as st
import chardet
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
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
    st.markdown("###### Upload CSV File and Get DataFrame with Repeated Questions")

    file = st.file_uploader("Upload here", type="csv")

    if file:
        # Detect the encoding of the file
        file_content = file.read()
        result = chardet.detect(file_content)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'
        file_content = io.StringIO(file_content.decode(encoding))

        df = pd.read_csv(file_content)
        df.columns = df.columns.str.strip()



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

                        # Load model
                        model = load_model()

                        # Encode sentences and compute similarity matrices
                        df['embedding'] = encode_sentences(model, df['السؤال'].tolist())
                        embeddings = np.array(df['embedding'].tolist())
                        similarity_matrix = cosine_similarity(embeddings)
                        distance_matrix = 1 - similarity_matrix
                        distance_matrix = np.clip(distance_matrix, a_min=0, a_max=None)

                        # Perform clustering
                        threshold = 0.9
                        eps = 1 - threshold
                        dbscan = DBSCAN(eps=eps, min_samples=8, metric='precomputed')
                        clusters = dbscan.fit_predict(distance_matrix)

                        # Add clusters to the DataFrame
                        df['category'] = clusters

                        # Handle noise points (outliers)
                        max_category = df['category'][df['category'] != -1].max() if len(
                            df[df['category'] != -1]) > 0 else 0
                        df['category'] = df['category'].apply(lambda x: max_category + 1 if x == -1 else x + 1)

                        # Drop the embedding column for clarity
                        df.drop('embedding', axis=1, inplace=True)
                        df_sorted = df.sort_values(by='category')

                        st.dataframe(df_sorted)
                        st.download_button(
                            label="Download Results",
                            data=df_sorted.to_csv(index=False).encode('utf-8'),
                            file_name="results.csv",
                            mime="text/csv",
                        )

                        st.session_state.button_clicked = False

if __name__ == "__main__":
    main()
