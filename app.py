# app.py - Music Recommender System 

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Music Recommender", layout="centered")

# ----------------- MUSIC RECOMMENDER -----------------
class MusicRecommender:
    def __init__(self):
        # Load dataset and similarity matrix
        self.df = pickle.load(open("df.pkl", "rb"))
        self.similarity = pickle.load(open("similarity.pkl", "rb"))

        # Safety reset index
        self.df = self.df.reset_index(drop=True)

    def recommend(self, song_name, top_k=5):
        if song_name not in self.df["song"].values:
            return pd.DataFrame()

        idx = self.df[self.df["song"] == song_name].index[0]
        sim_row = self.similarity[idx]

        # Fast top-k selection
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
st.title("🎵 Music Recommendation System")
st.caption("Content-Based Filtering using TF-IDF + Cosine Similarity")

song_list = recommender.df["song"].dropna().unique()
selected_song = st.selectbox("🎧 Select a song", song_list)

top_k = st.slider("🎯 Number of recommendations", 3, 10, 5)

if st.button("🚀 Show Recommendations"):
    result_df = recommender.recommend(selected_song, top_k)

    if not result_df.empty:
        st.subheader("🎼 Recommended Songs")

        # ----------- SAFE TABLE DISPLAY -----------
        st.table(result_df)

        # ----------------- SAFE VISUALIZATION -----------------
        st.subheader("📊 Similarity Score Visualization")
        chart_df = result_df[["Song", "Similarity Score"]].set_index("Song")
        st.bar_chart(chart_df)

        # ----------------- CSV EXPORT -----------------
        st.subheader("⬇️ Export Recommendations")

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Recommendations as CSV",
            data=csv,
            file_name="music_recommendations.csv",
            mime="text/csv"
        )

    else:
        st.warning("❌ No recommendations found.")
