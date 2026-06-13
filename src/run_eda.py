import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data import load_amazon_data, clean_data, split_by_product, download_and_cache_images

def main():
    # Setup directories
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    processed_dir = data_dir / "processed"
    image_dir = data_dir / "images"
    plots_dir = data_dir / "plots"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    reviews_df, meta_df = load_amazon_data("All_Beauty")
    
    # 2. Clean data
    reviews_df, meta_df = clean_data(reviews_df, meta_df)
    
    # 3. Perform exploratory data analysis calculations
    print("\n--- Exploratory Data Analysis ---")
    total_reviews = len(reviews_df)
    total_products = len(meta_df)
    unique_products_in_reviews = reviews_df['parent_asin'].nunique()
    
    print(f"Total reviews: {total_reviews}")
    print(f"Total products in metadata: {total_products}")
    print(f"Unique products in reviews: {unique_products_in_reviews}")
    
    # Rating distribution
    rating_counts = reviews_df['rating'].value_counts().sort_index()
    print("\nRating Distribution:")
    for rating, count in rating_counts.items():
        percentage = (count / total_reviews) * 100
        print(f"  {rating} Stars: {count} ({percentage:.2f}%)")
        
    # Price distribution
    prices = meta_df['price_cleaned'].dropna()
    print(f"\nPrice Summary:")
    print(f"  Mean Price: ${prices.mean():.2f}")
    print(f"  Median Price: ${prices.median():.2f}")
    print(f"  Min Price: ${prices.min():.2f}")
    print(f"  Max Price: ${prices.max():.2f}")
    
    # Review length distribution (in words)
    reviews_df['review_length'] = reviews_df['text_cleaned'].apply(lambda x: len(str(x).split()))
    lengths = reviews_df['review_length']
    print(f"\nReview Length Summary (words):")
    print(f"  Mean Length: {lengths.mean():.1f} words")
    print(f"  Median Length: {lengths.median():.1f} words")
    print(f"  Max Length: {lengths.max()} words")
    
    # Image availability
    products_with_images = meta_df['image_url'].notna().sum()
    print(f"\nImage Availability:")
    print(f"  Products with images: {products_with_images} ({products_with_images/total_products*100:.2f}%)")
    
    # Missing data report
    print("\nMissing Data Report (Metadata):")
    print(meta_df.isnull().sum())
    print("\nMissing Data Report (Reviews):")
    print(reviews_df.isnull().sum())
    
    # 4. Generate & Save Plots
    print("\nGenerating and saving plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Rating Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=reviews_df, x='rating', palette="viridis")
    plt.title("Distribution of Ratings in All_Beauty")
    plt.xlabel("Rating (Stars)")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(plots_dir / "rating_distribution.png", dpi=150)
    plt.close()
    
    # Plot 2: Price Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(prices, bins=30, kde=True, color='purple')
    plt.title("Distribution of Product Prices")
    plt.xlabel("Price ($)")
    plt.ylabel("Count")
    plt.xlim(0, 100) # zoom in on main range
    plt.tight_layout()
    plt.savefig(plots_dir / "price_distribution.png", dpi=150)
    plt.close()
    
    # Plot 3: Review Length Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=50, kde=True, color='teal')
    plt.title("Distribution of Review Word Counts")
    plt.xlabel("Word Count")
    plt.ylabel("Count")
    plt.xlim(0, 200) # zoom in on main range
    plt.tight_layout()
    plt.savefig(plots_dir / "review_length_distribution.png", dpi=150)
    plt.close()
    
    # 5. Split data by product (prevent leakage)
    (reviews_train, meta_train), (reviews_val, meta_val), (reviews_test, meta_test) = split_by_product(
        reviews_df, meta_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # Save processed splits
    print("Saving processed datasets (parquet format)...")
    reviews_train.to_parquet(processed_dir / "reviews_train.parquet", index=False)
    reviews_val.to_parquet(processed_dir / "reviews_val.parquet", index=False)
    reviews_test.to_parquet(processed_dir / "reviews_test.parquet", index=False)
    
    meta_train.to_parquet(processed_dir / "meta_train.parquet", index=False)
    meta_val.to_parquet(processed_dir / "meta_val.parquet", index=False)
    meta_test.to_parquet(processed_dir / "meta_test.parquet", index=False)
    
    # 6. Download sample of product images
    # We download 500 images as a sample for Milestone 0 to run quickly, but configurable.
    image_paths = download_and_cache_images(meta_df, image_dir, max_images=500)
    
    # Save the image paths mapping to json
    with open(processed_dir / "image_paths_map.json", "w") as f:
        json.dump(image_paths, f, indent=4)
        
    print("\nMilestone 0 Completed Successfully!")

if __name__ == "__main__":
    main()
