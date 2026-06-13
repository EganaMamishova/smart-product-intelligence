import os
import json
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import io
from PIL import Image
from tqdm import tqdm

def load_amazon_data(category="All_Beauty"):
    """
    Loads raw reviews and metadata from Hugging Face datasets for a given category.
    """
    print(f"Loading reviews for {category}...")
    reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}", trust_remote_code=True)
    reviews_df = pd.DataFrame(reviews_dataset['full'])
    
    print(f"Loading metadata for {category}...")
    meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", split="full", trust_remote_code=True)
    meta_df = pd.DataFrame(meta_dataset)
    
    return reviews_df, meta_df

def clean_data(reviews_df, meta_df):
    """
    Performs basic cleaning of reviews and metadata:
    - Handles missing values
    - Formats data types (e.g. price, ratings)
    - Extracts list elements from metadata where necessary
    """
    print("Cleaning metadata...")
    # Clean price: convert to float if possible, or NaN
    def clean_price(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        # If it's a string, try to parse
        x_str = str(x).replace('$', '').replace(',', '').strip()
        try:
            return float(x_str)
        except ValueError:
            return np.nan
            
    meta_df['price_cleaned'] = meta_df['price'].apply(clean_price)
    
    # Process images: extract the first image URL if it is a list of dicts/strings
    def extract_image_url(images_val):
        if not images_val or pd.isna(images_val):
            return None
            
        def first_string(val):
            if isinstance(val, list):
                if len(val) > 0:
                    first = val[0]
                    if isinstance(first, str):
                        return first
                    elif isinstance(first, dict) and 'large' in first:
                        return first['large']
            elif isinstance(val, str):
                return val
            return None

        if isinstance(images_val, dict):
            for key in ['large', 'hi_res', 'thumb']:
                if key in images_val:
                    res = first_string(images_val[key])
                    if res:
                        return res
            return None
            
        if isinstance(images_val, (list, np.ndarray)):
            return first_string(list(images_val))
            
        return None

    meta_df['image_url'] = meta_df['images'].apply(extract_image_url)
    
    # Clean descriptions (joining list of descriptions to single string)
    def clean_description(desc):
        if isinstance(desc, list):
            return " ".join([str(d) for d in desc])
        if pd.isna(desc):
            return ""
        return str(desc)
        
    meta_df['description_cleaned'] = meta_df['description'].apply(clean_description)
    
    print("Cleaning reviews...")
    # Convert ratings and helpful votes
    reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')
    vote_col = 'helpful_vote' if 'helpful_vote' in reviews_df.columns else 'helpful_votes'
    if vote_col in reviews_df.columns:
        reviews_df['helpful_votes'] = pd.to_numeric(reviews_df[vote_col], errors='coerce').fillna(0).astype(int)
    else:
        reviews_df['helpful_votes'] = 0
    reviews_df['text_cleaned'] = reviews_df['text'].fillna("")
    reviews_df['title_cleaned'] = reviews_df['title'].fillna("")
    
    return reviews_df, meta_df

def split_by_product(reviews_df, meta_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Splits the dataset by product (parent_asin) to prevent review-level leakage.
    Ensures that all reviews and metadata for a product fall in the same split.
    """
    print("Splitting data by product...")
    # We use parent_asin as the product identifier. If not present, we use asin.
    prod_col = 'parent_asin' if ('parent_asin' in meta_df.columns and 'parent_asin' in reviews_df.columns) else 'asin'
    
    # Get unique product IDs that exist in BOTH reviews and metadata
    unique_products = list(set(meta_df[prod_col].unique()) & set(reviews_df[prod_col].unique()))
    
    # Train, validation, test split on products
    train_prods, test_val_prods = train_test_split(
        unique_products, 
        train_size=train_ratio, 
        random_state=random_seed
    )
    
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    val_prods, test_prods = train_test_split(
        test_val_prods, 
        train_size=val_relative_ratio, 
        random_state=random_seed
    )
    
    train_prods_set = set(train_prods)
    val_prods_set = set(val_prods)
    test_prods_set = set(test_prods)
    
    # Split the metadata
    meta_train = meta_df[meta_df[prod_col].isin(train_prods_set)]
    meta_val = meta_df[meta_df[prod_col].isin(val_prods_set)]
    meta_test = meta_df[meta_df[prod_col].isin(test_prods_set)]
    
    # Split the reviews
    reviews_train = reviews_df[reviews_df[prod_col].isin(train_prods_set)]
    reviews_val = reviews_df[reviews_df[prod_col].isin(val_prods_set)]
    reviews_test = reviews_df[reviews_df[prod_col].isin(test_prods_set)]
    
    print(f"Split Summary (Products): Train={len(meta_train)}, Val={len(meta_val)}, Test={len(meta_test)}")
    print(f"Split Summary (Reviews): Train={len(reviews_train)}, Val={len(reviews_val)}, Test={len(reviews_test)}")
    
    return (reviews_train, meta_train), (reviews_val, meta_val), (reviews_test, meta_test)

def download_and_cache_images(meta_df, output_dir, max_images=5000):
    """
    Downloads and caches product images locally.
    Returns a mapping of product ID (asin or parent_asin) to local file path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter products that have valid image URLs
    valid_images = meta_df[meta_df['image_url'].notna() & (meta_df['image_url'] != "")]
    
    # Sample up to max_images
    if len(valid_images) > max_images:
        valid_images = valid_images.sample(n=max_images, random_state=42)
        
    print(f"Downloading {len(valid_images)} product images to {output_path}...")
    
    image_paths = {}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # Determine the ID column
    id_col = 'asin' if 'asin' in meta_df.columns else ('parent_asin' if 'parent_asin' in meta_df.columns else 'asin')
    
    for _, row in tqdm(valid_images.iterrows(), total=len(valid_images)):
        prod_id = row[id_col]
        url = row['image_url']
        ext = Path(url).suffix if Path(url).suffix in ['.jpg', '.jpeg', '.png'] else '.jpg'
        local_file = output_path / f"{prod_id}{ext}"
        
        # Check if already downloaded
        if local_file.exists():
            image_paths[prod_id] = str(local_file)
            continue
            
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                # Convert to RGB if needed (handles RGBA/CMYK/etc)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(local_file, "JPEG")
                image_paths[prod_id] = str(local_file)
            else:
                image_paths[prod_id] = None
        except Exception as e:
            image_paths[prod_id] = None
            
    return image_paths
