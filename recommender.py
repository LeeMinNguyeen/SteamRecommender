"""
Recommendation system functions for Steam games
Includes Content-based, Collaborative, and Hybrid filtering
"""
import numpy as np
import pandas as pd


def recommend_based_on_content(game_title, cosine_sim_matrix, games_df, n_recommendations=10):
    """
    Generates content-based recommendations for a given game.

    Args:
        game_title (str): The title of the game to get recommendations for.
        cosine_sim_matrix (np.ndarray): The precomputed content-based cosine similarity matrix.
        games_df (pd.DataFrame): The dataframe containing game titles and their indices.
        n_recommendations (int): The number of recommendations to return.

    Returns:
        list: A list of recommended game titles, or an error message if the game is not found.
    """
    try:
        # Find the index of the game that matches the title in the sampled dataframe
        game_index = games_df.index[games_df['title'] == game_title][0]
        # Get the position of this index in the dataframe's index list
        game_position = games_df.index.get_loc(game_index)
    except (IndexError, KeyError):
        return f"Game '{game_title}' not found in the games data for content-based filtering."

    # Get the similarity scores for this game with all other games
    similarity_scores = list(enumerate(cosine_sim_matrix[game_position]))

    # Sort the games based on the similarity scores
    sorted_similar_games = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar games (exclude the game itself)
    top_similar_games = [game for game in sorted_similar_games if game[0] != game_position][:n_recommendations]

    # Get the game titles and similarity scores of the recommended games
    recommended_games = [(games_df.iloc[idx]['title'], score) for idx, score in top_similar_games]

    return recommended_games


def recommend_based_on_collaborative(game_title, item_similarity_matrix, data, n_recommendations=10):
    """
    Generates item-based recommendations using collaborative filtering artifacts.

    Args:
        game_title (str): The title of the game to get recommendations for.
        item_similarity_matrix (np.ndarray): The precomputed item-item similarity matrix.
        data (pd.DataFrame): The dataframe containing game titles and their app_ids.
        n_recommendations (int): The number of recommendations to return.

    Returns:
        list: A list of recommended game titles, or an error message if the game is not found.
    """
    try:
        # Get the index corresponding to the item_similarity_matrix
        game_idx_in_filtered = data[data['title'] == game_title]['app_id'].cat.codes.iloc[0]
    except (IndexError, KeyError):
        return f"Game '{game_title}' not found in the filtered data for collaborative filtering."

    # Get the similarity scores for this game with all other games
    similarity_scores = list(enumerate(item_similarity_matrix[game_idx_in_filtered]))

    # Sort the games based on the similarity scores
    sorted_similar_games = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar games (exclude the game itself)
    top_similar_games = [game for game in sorted_similar_games if game[0] != game_idx_in_filtered][:n_recommendations]

    # Map the category codes back to app_ids to get titles
    app_id_map = dict(enumerate(data['app_id'].cat.categories))
    recommended_games = [(data[data['app_id'] == app_id_map[idx]]['title'].iloc[0], score) 
                         for idx, score in top_similar_games]

    return recommended_games


def recommend_hybrid(game_title, combined_cosine_sim_matrix, games_with_info_df, 
                     item_similarity_matrix, filtered_data_df, 
                     n_recommendations=10, content_weight=0.5, collaborative_weight=0.5):
    """
    Generates hybrid recommendations for a given game by combining content-based
    and collaborative filtering results.

    Args:
        game_title (str): The title of the game to get recommendations for.
        combined_cosine_sim_matrix: Content-based similarity matrix
        games_with_info_df: Games dataframe for content-based
        item_similarity_matrix: Collaborative similarity matrix
        filtered_data_df: Filtered data for collaborative
        n_recommendations (int): The total number of recommendations to return.
        content_weight (float): Weight for content-based recommendations (between 0 and 1).
        collaborative_weight (float): Weight for collaborative filtering recommendations.

    Returns:
        list: A list of unique recommended game titles with scores.
    """
    # Ensure weights sum to 1
    total_weight = content_weight + collaborative_weight
    if total_weight != 1.0 and total_weight != 0:
        content_weight /= total_weight
        collaborative_weight /= total_weight
    elif total_weight == 0:
        content_weight = 0.5
        collaborative_weight = 0.5

    # Get content-based recommendations
    content_recs = recommend_based_on_content(
        game_title,
        combined_cosine_sim_matrix,
        games_with_info_df,
        n_recommendations=int(n_recommendations * 2)
    )

    # Get collaborative filtering recommendations
    collaborative_recs = recommend_based_on_collaborative(
        game_title,
        item_similarity_matrix,
        filtered_data_df,
        n_recommendations=int(n_recommendations * 2)
    )

    # Handle cases where the game is not found
    game_found_content = isinstance(content_recs, list)
    game_found_collaborative = isinstance(collaborative_recs, list)

    if not game_found_content and not game_found_collaborative:
        return f"Game '{game_title}' not found in either content-based or collaborative filtering datasets."
    elif not game_found_content:
        combined_recommendations = collaborative_recs
    elif not game_found_collaborative:
        combined_recommendations = content_recs
    else:
        # Combine recommendations with weighted scores
        combined_dict = {}
        
        # Add content-based recommendations
        for title, score in content_recs:
            combined_dict[title] = combined_dict.get(title, 0) + (score * content_weight)
        
        # Add collaborative recommendations
        for title, score in collaborative_recs:
            combined_dict[title] = combined_dict.get(title, 0) + (score * collaborative_weight)
        
        # Sort by combined score
        combined_recommendations = sorted(combined_dict.items(), key=lambda x: x[1], reverse=True)

    # Remove the input game if it appears and return top N
    final_recommendations = [(title, score) for title, score in combined_recommendations 
                             if title != game_title][:n_recommendations]

    return final_recommendations


def get_available_games_for_recommendation(filtered_data_df, games_with_info_df):
    """
    Get the intersection of games available in both filtering methods
    
    Returns:
        list: Sorted list of game titles available for recommendations
    """
    collab_games = set(filtered_data_df['title'].unique()) if filtered_data_df is not None else set()
    content_games = set(games_with_info_df['title'].unique()) if games_with_info_df is not None else set()
    
    # Get games available in at least one method
    all_games = collab_games.union(content_games)
    
    return sorted(list(all_games))
