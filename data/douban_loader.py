import os
import pandas as pd
import csv

class DoubanLoader:
    def __init__(self, data_dir):
        """
        Loads Douban Dataset.
        Args:
            data_dir: Root directory containing 'douban_dataset(text information)'
        """
        self.data_dir = data_dir
        self.book_reviews_file = os.path.join(data_dir, "douban_dataset(text information)", "bookreviews_cleaned.txt")
        self.movie_reviews_file = os.path.join(data_dir, "douban_dataset(text information)", "moviereviews_cleaned.txt")
        self.music_reviews_file = os.path.join(data_dir, "douban_dataset(text information)", "musicreviews_cleaned.txt")

    def load_reviews(self, domain='book'):
        """
        Loads reviews for a specific domain.
        Args:
            domain: 'book', 'movie', or 'music'
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        if domain == 'book':
            file_path = self.book_reviews_file
        elif domain == 'movie':
            file_path = self.movie_reviews_file
        elif domain == 'music':
            file_path = self.music_reviews_file
        else:
            raise ValueError("Domain must be 'book', 'movie', or 'music'")
            
        if not os.path.exists(file_path):
            print(f"Warning: Douban file not found at {file_path}")
            return None
        
        print(f"Loading Douban {domain} reviews from {file_path}...")
        
        # Files are tab-separated with full quoting.
        try:
            df = pd.read_csv(
                file_path, 
                sep='\t', 
                quoting=csv.QUOTE_ALL, 
                on_bad_lines='skip',
                encoding='utf-8' # Assuming utf-8
            )
        except Exception as e:
            print(f"Error loading Douban data: {e}")
            return None
            
        return df

    def get_text_data(self, df):
        """
        Extracts user, item, rating, and text.
        """
        if df is None:
            return None
            
        # Standardize columns if possible
        # Header: "user_id" "book_id" "rating" "labels" "comment" "time" "ID"
        # Pandas likely stripped quotes from header if quoting=QUOTE_ALL is used correctly.
        
        # Rename col 'book_id' or similar to 'item_id'
        rename_map = {}
        for col in df.columns:
            if 'book_id' in col or 'movie_id' in col or 'music_id' in col:
                rename_map[col] = 'item_id'
            if 'comment' in col:
                rename_map[col] = 'text'
        
        df.rename(columns=rename_map, inplace=True)
        
        required_cols = ['user_id', 'item_id', 'rating', 'text']
        available_cols = [c for c in required_cols if c in df.columns]
        
        return df[available_cols]
