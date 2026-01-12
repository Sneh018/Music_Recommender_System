---------------------------------------
Author : Sneh Rojivadia
GitHub: https://github.com/Sneh018
---------------------------------------

# 🎵 Music Recommendation System

A scalable, content-based music recommendation system that delivers personalized song recommendations using natural language processing and machine learning techniques. The system analyzes song lyrics using TF-IDF vectorization and cosine similarity to identify semantically similar tracks and provides real-time recommendations through an interactive web interface.

This project demonstrates end-to-end machine learning workflow including data preprocessing, feature engineering, model optimization, performance tuning, and cloud deployment readiness.

---

## 🚀 Live Demo
- https://sneh-music-recommender-system.streamlit.app

---

## Key Features

- Content-based recommendation engine using TF-IDF + cosine similarity  
- Fast real-time inference using precomputed similarity matrices  
- Interactive similarity score visualization  
- Export recommendations to CSV  
- Clean and responsive Streamlit UI  
- Optimized memory usage using reduced dataset and float precision  
- Deployment-ready for free cloud platforms  

---

## System Architecture

User Interface (Streamlit)
↓
Precomputed Similarity Matrix (NumPy)
↓
TF-IDF Feature Space (Scikit-Learn)
↓
Processed Song Dataset (Pandas)

The similarity matrix is computed offline to avoid expensive computation during runtime, enabling sub-second recommendation latency.

---

## Dataset

- Source: Spotify lyrics dataset  
- Number of songs used: ~1,500  
- Features:
  - Song title  
  - Artist  
  - Lyrics text  

The dataset is intentionally reduced to balance performance, memory footprint, and cloud deployment constraints.

---

## Machine Learning Approach

### 1. Text Preprocessing
- Lowercasing  
- Removing special characters  
- Tokenization  
- Stop-word removal  

### 2. Feature Engineering
- TF-IDF Vectorization converts lyrics into numerical vectors  
- Max feature limit applied for dimensionality control  

### 3. Similarity Computation
- Cosine similarity measures closeness between song vectors  
- Similarity matrix is precomputed and stored for fast retrieval  

### 4. Optimization
- Reduced dataset size for deployment feasibility  
- Float16 precision for memory efficiency  
- Fast top-K retrieval using NumPy partitioning  

---

## Tech Stack

Programming Language: Python
Web Framework: Streamlit
Machine Learning: Scikit-learn
Data Processing: Pandas, NumPy
Visualization: Matplotlib

---

## Project Structure

Music_Recommender_System/
├── app.py
├── df.pkl
├── similarity.pkl
├── requirements.txt
├── README.md
└── LICENSE 

---

## Local Installation

1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/Music_Recommender_System.git
cd Music_Recommender_System

2. Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies 
pip install -r requirements.txt

4. Run the Application
streamlit run app.py

5. Open your browser and visit:
http://localhost:8501 

---

## License : 
This project is licensed under the MIT License — free to use, modify, and distribute.

---

## Future Improvements
- Search-based song lookup
- User favorite playlists
- Audio preview integration
- Hybrid recommendation (content + collaborative)
- Dockerized deployment
- Mobile-friendly UI
