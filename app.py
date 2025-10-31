"""
Steam Game Recommender System - Streamlit App
A comprehensive application for exploring data, visualizing EDA, and testing recommendation models
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils import load_all_data, clean_game_dataset, thresholds, threshold_filtering
from recommender import (recommend_based_on_content, recommend_based_on_collaborative, 
                         recommend_hybrid, get_available_games_for_recommendation)

# Page configuration
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #43a047;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .game-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .recommendation-item {
        background-color: #ffffff;
        border-left: 4px solid #1e88e5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all data and models with caching"""
    with st.spinner("Loading data and models..."):
        return load_all_data()


def show_home():
    """Display home page"""
    st.markdown('<h1 class="main-header">🎮 Steam Game Recommender System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Steam Game Recommender System!
    
    This application showcases a comprehensive recommendation system for Steam games using:
    - **Content-Based Filtering**: Recommends games based on game features (description, tags)
    - **Collaborative Filtering**: Recommends games based on user behavior patterns
    - **Hybrid Filtering**: Combines both approaches for better recommendations
    
    ---
    
    ### 📊 Navigate through the sections:
    
    #### 🔍 **Exploratory Data Analysis (EDA)**
    - Explore the Steam games dataset
    - View interactive visualizations
    - Understand data distributions and patterns
    
    #### 🧹 **Data Preprocessing**
    - See how the data was cleaned and prepared
    - View filtering steps and threshold calculations
    - Understand the preprocessing pipeline
    
    #### 🤖 **Test Recommendation Models**
    - Try out the recommendation system
    - Select a game and get personalized recommendations
    - Compare different filtering approaches
    
    ---
    
    ### 🚀 Get Started
    Use the **sidebar** to navigate to different sections!
    """)
    
    # Display some quick stats if data is loaded
    data = load_data()
    if data['games'] is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Games", f"{len(data['games']):,}")
        with col2:
            if data['filtered_data'] is not None:
                st.metric("Games in Collaborative Model", f"{data['filtered_data']['title'].nunique():,}")
        with col3:
            if data['games_with_info'] is not None:
                st.metric("Games in Content-Based Model", f"{len(data['games_with_info']):,}")
        with col4:
            if data['games_metadata'] is not None:
                st.metric("Games with Metadata", f"{len(data['games_metadata']):,}")


def show_eda():
    """Display Exploratory Data Analysis section"""
    st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    data = load_data()
    games_df = data['games']
    games_metadata = data['games_metadata']
    
    if games_df is None:
        st.error("Unable to load games data. Please check if the data files exist.")
        return
    
    # Dataset Overview
    st.markdown('<h2 class="section-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Games Data", "Statistics", "Data Info"])
    
    with tab1:
        st.subheader("Sample Games Data")
        st.dataframe(games_df.head(20), use_container_width=True)
        
        st.subheader("Dataset Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Rows:** {games_df.shape[0]:,}")
        with col2:
            st.info(f"**Columns:** {games_df.shape[1]}")
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(games_df.describe(), use_container_width=True)
    
    with tab3:
        st.subheader("Data Types and Missing Values")
        info_df = pd.DataFrame({
            'Column': games_df.columns,
            'Data Type': games_df.dtypes,
            'Missing Values': games_df.isna().sum(),
            'Missing %': (games_df.isna().sum() / len(games_df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)
    
    # Visualizations
    st.markdown('<h2 class="section-header">📈 Data Visualizations</h2>', unsafe_allow_html=True)
    
    # Price Distribution
    st.subheader("💰 Price Distribution")
    fig_price = px.histogram(
        games_df, 
        x='price_original', 
        nbins=50,
        title='Distribution of Game Prices',
        labels={'price_original': 'Price (USD)', 'count': 'Number of Games'},
        color_discrete_sequence=['#1e88e5']
    )
    fig_price.update_layout(height=400)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Rating Distribution
    st.subheader("⭐ Rating Distribution")
    rating_counts = games_df['rating'].value_counts()
    fig_rating = px.bar(
        x=rating_counts.index, 
        y=rating_counts.values,
        title='Distribution of Game Ratings',
        labels={'x': 'Rating', 'y': 'Number of Games'},
        color=rating_counts.values,
        color_continuous_scale='Viridis'
    )
    fig_rating.update_layout(height=400)
    st.plotly_chart(fig_rating, use_container_width=True)
    
    # Platform Distribution
    st.subheader("🖥️ Platform Support")
    col1, col2, col3 = st.columns(3)
    with col1:
        win_count = games_df['win'].sum()
        st.metric("Windows Support", f"{win_count:,}", f"{win_count/len(games_df)*100:.1f}%")
    with col2:
        mac_count = games_df['mac'].sum()
        st.metric("Mac Support", f"{mac_count:,}", f"{mac_count/len(games_df)*100:.1f}%")
    with col3:
        linux_count = games_df['linux'].sum()
        st.metric("Linux Support", f"{linux_count:,}", f"{linux_count/len(games_df)*100:.1f}%")
    
    # User Reviews Distribution
    st.subheader("👥 User Reviews Distribution")
    fig_reviews = px.box(
        games_df, 
        y='user_reviews',
        title='Distribution of User Reviews',
        labels={'user_reviews': 'Number of Reviews'},
        color_discrete_sequence=['#43a047']
    )
    fig_reviews.update_layout(height=400)
    st.plotly_chart(fig_reviews, use_container_width=True)
    
    # Positive Ratio vs Price
    st.subheader("📊 Positive Ratio vs Price")
    fig_scatter = px.scatter(
        games_df.sample(min(5000, len(games_df))),
        x='price_original',
        y='positive_ratio',
        color='rating',
        size='user_reviews',
        hover_data=['title'],
        title='Game Price vs Positive Rating Ratio',
        labels={'price_original': 'Price (USD)', 'positive_ratio': 'Positive Ratio (%)'},
        opacity=0.6
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Top Games by Reviews
    st.subheader("🏆 Top 20 Games by User Reviews")
    top_games = games_df.nlargest(20, 'user_reviews')[['title', 'user_reviews', 'positive_ratio', 'price_original']]
    fig_top = px.bar(
        top_games,
        x='user_reviews',
        y='title',
        orientation='h',
        title='Most Reviewed Games',
        labels={'user_reviews': 'Number of Reviews', 'title': 'Game Title'},
        color='positive_ratio',
        color_continuous_scale='RdYlGn'
    )
    fig_top.update_layout(height=600)
    st.plotly_chart(fig_top, use_container_width=True)


def show_preprocessing():
    """Display Data Preprocessing section"""
    st.markdown('<h1 class="main-header">🧹 Data Preprocessing</h1>', unsafe_allow_html=True)
    
    data = load_data()
    games_df = data['games']
    
    if games_df is None:
        st.error("Unable to load games data.")
        return
    
    st.markdown("""
    This section demonstrates the data cleaning and preprocessing steps applied to prepare 
    the dataset for the recommendation models.
    """)
    
    # Game Dataset Cleaning
    st.markdown('<h2 class="section-header">🎮 Game Dataset Cleaning</h2>', unsafe_allow_html=True)
    
    st.write("### Original Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Games Count", f"{len(games_df):,}")
    with col2:
        st.metric("Original Features", games_df.shape[1])
    
    st.write("### Cleaning Steps")
    
    # Step 1: Price Filtering
    with st.expander("Step 1: Price Filtering", expanded=True):
        st.write("""
        **Filter Criteria:**
        - Free games (price = $0.00) OR
        - Paid games between $2.99 and $70.00
        
        This removes extremely cheap games and unreasonably expensive outliers.
        """)
        
        clean_price_df = games_df[
            ((games_df['price_original']) == 0.00) | 
            ((games_df['price_original'] >= 2.99) & (games_df['price_original'] <= 70.00))
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Games After Price Filter", f"{len(clean_price_df):,}")
        with col2:
            st.metric("Games Removed", f"{len(games_df) - len(clean_price_df):,}")
    
    # Step 2: Platform Filtering
    with st.expander("Step 2: Platform Filtering", expanded=True):
        st.write("""
        **Filter Criteria:**
        - Only games with Windows support
        
        Focuses on the most widely supported platform.
        """)
        
        clean_platform_df = clean_price_df[clean_price_df['win']]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Games After Platform Filter", f"{len(clean_platform_df):,}")
        with col2:
            st.metric("Games Removed", f"{len(clean_price_df) - len(clean_platform_df):,}")
    
    # Step 3: Rating Filtering
    with st.expander("Step 3: Rating Filtering", expanded=True):
        st.write("""
        **Filter Criteria:**
        - Remove games with 'Overwhelmingly Negative' or 'Very Negative' ratings
        
        Filters out poorly received games to improve recommendation quality.
        """)
        
        ratings_to_filter = ['Overwhelmingly Negative', 'Very Negative']
        clean_final_df = clean_platform_df[~clean_platform_df['rating'].isin(ratings_to_filter)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Games After Rating Filter", f"{len(clean_final_df):,}")
        with col2:
            st.metric("Games Removed", f"{len(clean_platform_df) - len(clean_final_df):,}")
    
    # Final Summary
    st.write("### Final Dataset Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Count", f"{len(games_df):,}")
    with col2:
        st.metric("Final Count", f"{len(clean_final_df):,}")
    with col3:
        reduction = (1 - len(clean_final_df)/len(games_df)) * 100
        st.metric("Reduction", f"{reduction:.1f}%")
    
    # Outlier Detection
    st.markdown('<h2 class="section-header">📉 Outlier Detection & Handling</h2>', unsafe_allow_html=True)
    
    st.write("""
    The Interquartile Range (IQR) method was used to detect and handle outliers in user-related data.
    
    **IQR Formula:**
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1
    - Lower Threshold = Q1 - 1.5 × IQR
    - Upper Threshold = Q3 + 1.5 × IQR
    """)
    
    # Show boxplot for user reviews
    st.write("### User Reviews Distribution (Before Outlier Removal)")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(x=games_df['user_reviews'], color="skyblue", ax=ax)
    ax.set_title('Number of User Reviews - Boxplot')
    ax.set_xlabel('Number of Reviews')
    st.pyplot(fig)
    plt.close()
    
    # Calculate and display thresholds
    min_threshold, max_threshold = thresholds(games_df, 'user_reviews')
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Lower Threshold:** {min_threshold:.2f}")
    with col2:
        st.info(f"**Upper Threshold:** {max_threshold:.2f}")


def show_model_testing():
    """Display Model Testing section"""
    st.markdown('<h1 class="main-header">🤖 Test Recommendation Models</h1>', unsafe_allow_html=True)
    
    data = load_data()
    
    # Check if models are loaded
    if data['combined_cosine_sim_matrix'] is None or data['item_similarity_matrix'] is None:
        st.error("""
        Unable to load model files. Please ensure the following files exist:
        - Models/Collaborative/filtered_data.pkl
        - Models/Collaborative/item_similarity_matrix.pkl
        - Models/Content-based/games_with_info.pkl
        - Models/Content-based/combined_cosine_sim_matrix.pkl
        """)
        return
    
    # Get available games
    available_games = get_available_games_for_recommendation(
        data['filtered_data'],
        data['games_with_info']
    )
    
    if not available_games:
        st.error("No games available for recommendations.")
        return
    
    st.markdown("""
    Select a game you enjoy, and our recommendation system will suggest similar games 
    using different filtering approaches!
    """)
    
    # Model selection and configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_game = st.selectbox(
            "🎮 Select a game you like:",
            options=available_games,
            index=0
        )
    
    with col2:
        n_recommendations = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
    
    # Model type selection
    st.markdown("### Choose Recommendation Method:")
    model_type = st.radio(
        "Select filtering approach:",
        ["Hybrid (Recommended)", "Content-Based", "Collaborative"],
        horizontal=True
    )
    
    # Hybrid method configuration
    content_weight = 0.5
    collaborative_weight = 0.5
    
    if model_type == "Hybrid (Recommended)":
        st.markdown("#### Adjust Hybrid Weights:")
        content_weight = st.slider(
            "Content-Based Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        collaborative_weight = 1.0 - content_weight
        st.info(f"Collaborative Weight: {collaborative_weight:.1f}")
    
    # Generate recommendations button
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            try:
                if model_type == "Content-Based":
                    recommendations = recommend_based_on_content(
                        selected_game,
                        data['combined_cosine_sim_matrix'],
                        data['games_with_info'],
                        n_recommendations
                    )
                elif model_type == "Collaborative":
                    recommendations = recommend_based_on_collaborative(
                        selected_game,
                        data['item_similarity_matrix'],
                        data['filtered_data'],
                        n_recommendations
                    )
                else:  # Hybrid
                    recommendations = recommend_hybrid(
                        selected_game,
                        data['combined_cosine_sim_matrix'],
                        data['games_with_info'],
                        data['item_similarity_matrix'],
                        data['filtered_data'],
                        n_recommendations,
                        content_weight,
                        collaborative_weight
                    )
                
                # Display recommendations
                if isinstance(recommendations, str):
                    st.warning(recommendations)
                else:
                    st.markdown(f"""
                    <div class="game-card">
                        <h2>🎮 You selected: {selected_game}</h2>
                        <p>Based on your selection, here are {len(recommendations)} recommended games:</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 🎯 Recommended Games:")
                    
                    # Display recommendations in a nice format
                    for i, (title, score) in enumerate(recommendations, 1):
                        # Get game details if available
                        game_info = data['games'][data['games']['title'] == title]
                        
                        if not game_info.empty:
                            game_details = game_info.iloc[0]
                            price = game_details['price_original']
                            rating = game_details['rating']
                            positive_ratio = game_details['positive_ratio']
                            
                            st.markdown(f"""
                            <div class="recommendation-item">
                                <h3>{i}. {title}</h3>
                                <p>
                                    <strong>Similarity Score:</strong> {score:.3f} | 
                                    <strong>Rating:</strong> {rating} ({positive_ratio}% positive) | 
                                    <strong>Price:</strong> ${price:.2f}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="recommendation-item">
                                <h3>{i}. {title}</h3>
                                <p><strong>Similarity Score:</strong> {score:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Show visualization of similarity scores
                    st.markdown("### 📊 Similarity Scores Visualization")
                    fig = px.bar(
                        x=[title for title, _ in recommendations],
                        y=[score for _, score in recommendations],
                        labels={'x': 'Game Title', 'y': 'Similarity Score'},
                        title='Recommendation Similarity Scores',
                        color=[score for _, score in recommendations],
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=500,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)


def main():
    """Main application function"""
    # Sidebar navigation
    st.sidebar.title("Group 5")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Go to:",
        ["🏠 Home", "📊 EDA", "🧹 Data Preprocessing", "🤖 Test Models"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This Steam Game Recommender System uses:
    - **Content-Based Filtering**
    - **Collaborative Filtering**
    - **Hybrid Approach**
    """)
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home()
    elif page == "📊 EDA":
        show_eda()
    elif page == "🧹 Data Preprocessing":
        show_preprocessing()
    elif page == "🤖 Test Models":
        show_model_testing()


if __name__ == "__main__":
    main()
