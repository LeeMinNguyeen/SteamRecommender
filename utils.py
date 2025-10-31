"""
Utility functions for loading data and models
"""
import pickle
import pandas as pd
import os
import streamlit as st


@st.cache_data
def load_pickle(filepath):
    """Load pickle file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading {filepath}: {str(e)}")
        return None


@st.cache_data
def load_csv(filepath):
    """Load CSV file"""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading {filepath}: {str(e)}")
        return None


@st.cache_data
def load_json(filepath):
    """Load JSON file"""
    try:
        return pd.read_json(filepath, lines=True)
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading {filepath}: {str(e)}")
        return None


def load_all_data():
    """Load all necessary data and models"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    data = {
        # CSV files
        'games': load_csv(os.path.join(base_path, 'Data', 'games.csv')),
        'games_metadata': load_json(os.path.join(base_path, 'Data', 'games_metadata.json')),
        
        # Collaborative filtering models
        'filtered_data': load_pickle(os.path.join(base_path, 'Models', 'Collaborative', 'filtered_data.pkl')),
        'user_latent_factors': load_pickle(os.path.join(base_path, 'Models', 'Collaborative', 'user_latent_factors.pkl')),
        'game_latent_factors': load_pickle(os.path.join(base_path, 'Models', 'Collaborative', 'game_latent_factors.pkl')),
        'item_similarity_matrix': load_pickle(os.path.join(base_path, 'Models', 'Collaborative', 'item_similarity_matrix.pkl')),
        
        # Content-based filtering models
        'games_with_info': load_pickle(os.path.join(base_path, 'Models', 'Content-based', 'games_with_info.pkl')),
        'combined_cosine_sim_matrix': load_pickle(os.path.join(base_path, 'Models', 'Content-based', 'combined_cosine_sim_matrix.pkl')),
    }
    
    return data


def clean_game_dataset(game_df):
    """Clean game dataset based on preprocessing steps from notebook"""
    # Price
    clean_game_price_df = game_df[
        ((game_df['price_original']) == 0.00) | ((game_df['price_original'] >= 2.99) & (game_df['price_original'] <= 70.00))
    ]
    
    # Platform
    clean_game_platform_df = clean_game_price_df[clean_game_price_df['win']]
    
    # Ratings
    ratings_to_filter = ['Overwhelmingly Negative', 'Very Negative']
    clean_game_ratings_df = clean_game_platform_df[
        ~clean_game_platform_df['rating'].isin(ratings_to_filter)
    ]
    
    return clean_game_ratings_df


def thresholds(dataframe, column_name: str, moderate_outlier=True):
    """Calculate IQR thresholds for outlier detection"""
    Q1 = dataframe[column_name].quantile(.25)
    Q3 = dataframe[column_name].quantile(.75)
    
    # Interquartile Range
    IQR = Q3 - Q1
    
    # 1.5 for detecting moderate outliers else 3 for extreme and severe outliers
    multiplier = 1.5 if moderate_outlier else 3
    left_threshold = Q1 - multiplier * IQR
    right_threshold = Q3 + multiplier * IQR
    
    return left_threshold, right_threshold


def threshold_filtering(dataframe, column_name: str, min_threshold: float, max_threshold: float):
    """Filter dataframe based on column thresholds"""
    clean_dataframe = dataframe[
        (dataframe[column_name] > min_threshold) &
        (dataframe[column_name] <= max_threshold)
    ]
    return clean_dataframe
