# Steam Game Recommender System

A comprehensive machine learning-based recommendation system for Steam games with an interactive Streamlit web application. This project implements Content-Based Filtering, Collaborative Filtering, and Hybrid approaches to provide personalized game recommendations.

## Overview

This project builds a complete recommendation system for Steam games using machine learning techniques. The system analyzes game features (description, tags, genres) and user behavior patterns to suggest games that players might enjoy.

The project includes:
- **Jupyter Notebook** with complete data analysis, preprocessing, and model training
- **Streamlit Web App** for interactive data exploration and testing recommendations
- **Three Recommendation Engines**: Content-Based, Collaborative, and Hybrid


#### Model Testing
- **Interactive Interface**:
  - Searchable game dropdown (thousands of games)
  - Adjustable number of recommendations (5-20)
  - Three filtering methods to compare
- **Recommendation Methods**:
  - **Hybrid (Recommended)**: Combines both approaches with adjustable weights
  - **Content-Based**: Based on game features
  - **Collaborative**: Based on user behavior patterns
- **Rich Results Display**:
  - Game recommendations with similarity scores
  - Game details (price, rating, positive ratio)
  - Interactive bar chart visualization
  - Beautiful card-based layout

---

## Installation

### Data and modelfiles
Download and put these files in their respective folders (check Project Structure):
- Data: [Game Recommendations on Steam](https://www.kaggle.com/antonkozyriev/game-recommendations-on-steam)
- Models: [Checkpoints](https://drive.google.com/drive/folders/1IANB7DCRcufZYKUzXh2dLBLUr5Znyyr5)

### Prerequisites
- Python 3.12 or higher
- Git (optional, for cloning)
- Windows OS (or adapt commands for Mac/Linux)

### Step 1: Clone or Download
```bash
git clone https://github.com/LeeMinNguyeen/SteamRecommender.git
cd SteamRecommender
```

### Step 2: Set Up Virtual Environment
The virtual environment is already created. Activate it:
```powershell
.venv\Scripts\activate
```

If you need to create it from scratch:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Usage

```powershell
# Navigate to project directory
cd E:\Project\SteamRecommender

# Activate virtual environment
.venv\Scripts\activate

# Run the Streamlit app
streamlit run app.py
```

### Accessing the App

The app will automatically open in your default browser at:
**http://localhost:8501**

If it doesn't open automatically, manually navigate to the URL above.

### Using the Application

1. **Navigate**: Use the sidebar to switch between sections
2. **Explore EDA**: View interactive visualizations and data statistics
3. **Review Preprocessing**: See how data was cleaned step-by-step
4. **Test Recommendations**:
   - Select a game from the dropdown
   - Choose number of recommendations (5-20)
   - Pick a filtering method (Hybrid recommended)
   - Click "Get Recommendations"
   - View results with similarity scores and game details

### Stopping the App

Press `Ctrl + C` in the terminal window to stop the server.

---

## Project Structure

```
SteamRecommender/
├── app.py                          # Main Streamlit application
├── utils.py                        # Utility functions for data loading
├── recommender.py                  # Recommendation algorithms
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .venv/                          # Virtual environment (ignored in git)
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── Data/
│   ├── games.csv                   # Games dataset (~50k games)
│   ├── games_metadata.json         # Game metadata (descriptions, tags)
│   ├── users.csv                   # User data (large file)
│   └── recommendations.csv         # User-game recommendations (large file)
├── Models/
│   ├── Collaborative/
│   │   ├── filtered_data.pkl       # Preprocessed data for collaborative filtering
│   │   ├── user_latent_factors.pkl # User latent factors from SVD
│   │   ├── game_latent_factors.pkl # Game latent factors from SVD
│   │   └── item_similarity_matrix.pkl # Item-item similarity matrix
│   └── Content-based/
│       ├── games_with_info.pkl     # Games with metadata
│       └── combined_cosine_sim_matrix.pkl # Content similarity matrix
└── JupyterNotebook/
    └── RCM.ipynb                   # Original analysis and model training notebook
```

---

## Recommendation Methods

### 1. Content-Based Filtering

**How it works:**
- Analyzes game features: descriptions, tags, genres
- Uses **TF-IDF** (Term Frequency-Inverse Document Frequency) for text descriptions
- Uses **Multi-Label Binarizer** for game tags
- Combines features into a unified matrix
- Calculates **Cosine Similarity** between games
- Recommends games with similar features

**Best for:**
- Finding games with similar themes/gameplay
- Cold start problem (new users)
- Explaining why games are recommended

**Example:** If you like "Dark Souls" → Recommends "Bloodborne" (similar dark fantasy, challenging gameplay)

### 2. Collaborative Filtering

**How it works:**
- Analyzes user behavior and interaction patterns
- Uses **SVD** (Singular Value Decomposition) for dimensionality reduction
- Creates latent factor representations of users and items
- Calculates item-item similarity based on user interactions
- Recommends games that similar users enjoyed

**Best for:**
- Discovering unexpected recommendations
- Leveraging community preferences
- Finding hidden gems

**Example:** If users who played "Portal" also enjoyed "The Talos Principle" → Recommends "The Talos Principle"

### 3. Hybrid Filtering (Recommended)

**How it works:**
- Combines Content-Based and Collaborative approaches
- Uses weighted scoring (default: 50% content, 50% collaborative)
- Adjustable weights via slider in the app
- Normalizes and merges similarity scores
- Provides more robust recommendations

**Best for:**
- Most accurate recommendations
- Balancing multiple factors
- Handling edge cases (games in only one dataset)

**Example:** Combines feature similarity AND community behavior for well-rounded suggestions

---

## Dataset

### Source
Dataset from Kaggle: [Game Recommendations on Steam](https://www.kaggle.com/antonkozyriev/game-recommendations-on-steam)

### Dataset Statistics

- **Total Games**: ~50,000 Steam games
- **Users**: Thousands of Steam users
- **Recommendations**: Millions of user-game interactions

### Game Features

| Feature | Description |
|---------|-------------|
| `app_id` | Unique game identifier |
| `title` | Game name |
| `date_release` | Release date |
| `win/mac/linux` | Platform support |
| `rating` | User rating category |
| `positive_ratio` | Percentage of positive reviews |
| `user_reviews` | Number of user reviews |
| `price_original` | Original price in USD |
| `price_final` | Current price (after discounts) |
| `discount` | Discount percentage |
| `steam_deck` | Steam Deck compatibility |

### Metadata Features
- `description` - Game description text
- `tags` - Community-assigned tags
- `genres` - Game genres

### Data Preprocessing Applied

1. **Price Filtering**: Keep free games OR games priced $2.99-$70.00
2. **Platform Filtering**: Keep only Windows-supported games
3. **Rating Filtering**: Remove "Overwhelmingly Negative" and "Very Negative" ratings
4. **Outlier Removal**: IQR method for user reviews and hours played
5. **User Filtering**: Minimum 5 reviews per user
6. **Game Filtering**: Minimum 20 reviews per game

**Result**: ~5% of original data retained for high-quality recommendations

---

## Technologies

### Core Technologies
- **Python 3.12**: Primary programming language
- **Streamlit 1.39**: Web application framework
- **Pandas 2.2**: Data manipulation and analysis
- **NumPy 1.26**: Numerical computing

### Machine Learning
- **Scikit-learn 1.5**: 
  - TF-IDF Vectorization
  - SVD (Singular Value Decomposition)
  - Cosine Similarity
  - Multi-Label Binarizer
- **SciPy 1.14**: Sparse matrix operations

### Visualization
- **Matplotlib 3.9**: Static visualizations
- **Seaborn 0.13**: Statistical visualizations
- **Plotly 5.24**: Interactive charts and graphs

### Development Tools
- **Jupyter Notebook**: Data exploration and model development
- **Git**: Version control
- **Virtual Environment**: Dependency isolation

## Performance Metrics

### Model Performance
- **Content-Based**: Fast recommendations (~0.1s per query)
- **Collaborative**: Moderate speed (~0.2s per query)
- **Hybrid**: Slightly slower (~0.3s per query) but more accurate

### Dataset Coverage
- **Content-Based Model**: ~5% of games (sampled for memory efficiency)
- **Collaborative Model**: Games with 20+ reviews
- **Hybrid**: Union of both datasets

### Sparsity
- Original user-item matrix: >99% sparse
- After filtering: Still highly sparse (realistic for recommendation systems)