'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''
import ast
import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv('data/prediction_model_03.csv')
    genres_df = pd.read_csv('data/Genres.csv')
    return model_pred_df, genres_df

def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    # Create a list of unique genres
    genre_list = genres_df['genre'].tolist()

    # Initialize Dictionaries
    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}

    # Process the rows in the Dataframe

    for _, row in model_pred_df.iterrows():
        actual_genres = ast.literal_eval(row['actual genres']) if isinstance(row['actual genres'], str) else [row['actual genres']]
        predicted_genre = row['predicted']
        is_correct = row['correct?']

        if is_correct:
            genre_tp_counts[predicted_genre] += 1
        else:
            genre_fp_counts[predicted_genre] += 1
        
        for genre in actual_genres:
            genre_true_counts[genre] += 1
    
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts


