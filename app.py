# app.py - Music Recommender System

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Music Recommender", layout="wide")

# ----------------- MUSIC RECOMMENDER -----------------
class MusicRecommender:
    def __init__(self):
        # Load FULL dataset (precomputed)
        self.df = pickle.load(open("df.pkl", "rb"))
        self.similarity = pickle.load(open("similarity.pkl", "rb"))

        # Optional safety reset index
        self.df = self.df.reset_index(drop=True)

    def recommend(self, song_name, top_k=5):
        if song_name not in self.df["song"].values:
            return pd.DataFrame()

        idx = self.df[self.df["song"] == song_name].index[0]
        sim_row = self.similarity[idx]

        # Fast top-k selection this avoids full sorting
        top_indices = np.argpartition(sim_row, -top_k-1)[-top_k-1:]
        top_indices = top_indices[top_indices != idx]
        top_indices = top_indices[np.argsort(sim_row[top_indices])[::-1]]

        recommendations = []

        for i in top_indices[:top_k]:
            recommendations.append({
                "Song": self.df.iloc[i]["song"],
                "Artist": self.df.iloc[i]["artist"],
                "Similarity Score": round(float(sim_row[i]), 4)
            })

        return pd.DataFrame(recommendations)


# ----------------- CACHE MODEL -----------------
@st.cache_resource
def load_recommender():
    return MusicRecommender()

recommender = load_recommender()

# ----------------- UI -----------------
st.title("🎵 Music Recommendation System (Full Dataset)")
st.caption("Content-Based Filtering using TF-IDF + Cosine Similarity")

song_list = recommender.df["song"].dropna().unique()
selected_song = st.selectbox("🎧 Select a song", song_list)

top_k = st.slider("🎯 Number of recommendations", 3, 10, 5)

if st.button("🚀 Show Recommendations"):
    result_df = recommender.recommend(selected_song, top_k)

    if not result_df.empty:
        st.subheader("🎼 Recommended Songs")

        # ----------- TABLE DISPLAY -----------
        st.dataframe(
            result_df,
            use_container_width=True
        )

        # ----------------- VISUALIZATION -----------------
        st.subheader("📊 Similarity Score Visualization")

        fig, ax = plt.subplots()
        ax.barh(
            result_df["Song"][::-1],
            result_df["Similarity Score"][::-1]
        )
        ax.set_xlabel("Cosine Similarity Score")
        ax.set_title("Top-K Similar Songs")

        st.pyplot(fig)

        # ----------------- CSV EXPORT -----------------
        st.subheader("Export Recommendations :")

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Recommendations as CSV",
            data=csv,
            file_name="music_recommendations.csv",
            mime="text/csv"
        )

    else:
        st.warning("❌ No recommendations found.")
