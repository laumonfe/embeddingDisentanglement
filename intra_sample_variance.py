import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, pdist
import os
from itertools import combinations

def load_embeddings(emb_path):
    """Load embeddings from .npy file"""
    if os.path.exists(emb_path):
        return np.load(emb_path, allow_pickle=True)
    else:
        print(f"File not found: {emb_path}")
        return None

def calculate_intra_sample_metrics(embeddings, df, metric='cosine'):
    """
    Calculate intra-sample distance metrics for embeddings.
    
    Args:
        embeddings: Array of embedding dictionaries with 'idx', 'desc_idx', and 'embedding'
        df: DataFrame with item_idx and desc_idx
        metric: 'cosine' or 'euclidean'
    
    Returns:
        DataFrame with metrics per item_idx
    """
    results = []
    
    # Group embeddings by item_idx
    embedding_dict = {}
    for emb in embeddings:
        key = emb['idx']
        if key not in embedding_dict:
            embedding_dict[key] = []
        embedding_dict[key].append(emb['embedding'])
    
    for item_idx, emb_list in embedding_dict.items():
        if len(emb_list) < 2:
            # Skip items with only one embedding
            continue
        
        emb_array = np.array(emb_list)
        distances = pdist(emb_array, metric='cosine')

        
        # Calculate metrics
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        std_distance = np.std(distances)
        
        # Distance to centroid
        centroid = np.mean(emb_array, axis=0)
        centroid_distances = [cosine(emb, centroid) for emb in emb_array]
        mean_centroid_dist = np.mean(centroid_distances)
        
        results.append({
            'item_idx': item_idx,
            'num_embeddings': len(emb_list),
            'avg_pairwise_distance': avg_distance,
            'max_pairwise_distance': max_distance,
            'std_pairwise_distance': std_distance,
            'mean_centroid_distance': mean_centroid_dist,
            'metric': metric
        })
    
    return pd.DataFrame(results)

def compare_all_models():
    """Compare intra-sample distances across all models"""
    
    # Load metadata
    csv_path = os.path.join("visualizations", "visualization_explorer", "static", "test_metadata.csv")
    df = pd.read_csv(csv_path)
    
    models = [
        ("baseline", "default", "text", "cosine"),
        ("finetuned", "default", "text", "cosine"),
        ("finetuned", "grouped", "text", "cosine"),
        ("disentangled", "default", "text", "cosine"),
        ("disentangled", "grouped", "text", "cosine"),
    ]
    
    all_results = []
    
    for model_kind, data_type, embedding_type, metric in models:
        print(f"\nProcessing {model_kind}/{data_type}/{embedding_type}...")
        
        # Construct paths
        if model_kind == "baseline":
            emb_dir = os.path.join("data", "embeddings", "baseline_clip-ViT-B-32-multilingual-v1")
            if embedding_type == "text":
                emb_path = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_baseline.npy")

        else:
            emb_dir = os.path.join("data", "embeddings", f"{model_kind}_{data_type}_clip-ViT-B-32-multilingual-v1")
            if embedding_type == "text":
                emb_path = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}_{data_type}.npy")
        
        # Load embeddings
        embeddings = load_embeddings(emb_path)
        if embeddings is None:
            continue
        
        # Filter for test split
        test_df = df[df["split"] == "test"]
        test_keys = set(zip(test_df["item_idx"], test_df["desc_idx"]))
        test_embeddings = [e for e in embeddings if (e['idx'], e['desc_idx']) in test_keys]
        
        # Calculate metrics
        metrics_df = calculate_intra_sample_metrics(test_embeddings, test_df, metric=metric)
        metrics_df['model'] = model_kind
        metrics_df['data_type'] = data_type
        metrics_df['embedding_type'] = embedding_type
        
        all_results.append(metrics_df)
        
        # Print summary statistics
        print(f"  Items analyzed: {len(metrics_df)}")
        print(f"  Mean avg pairwise distance: {metrics_df['avg_pairwise_distance'].mean():.4f}")
        print(f"  Mean max pairwise distance: {metrics_df['max_pairwise_distance'].mean():.4f}")
        print(f"  Mean centroid distance: {metrics_df['mean_centroid_distance'].mean():.4f}")
    
    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)
    # Create output directory if it doesn't exist
    output_dir = os.path.join("output", "intra_class")
    os.makedirs(output_dir, exist_ok=True)
    # Save results
    output_path = os.path.join(output_dir, "intra_sample_distances.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Create summary comparison
    summary = final_df.groupby(['model', 'data_type', 'embedding_type']).agg({
        'avg_pairwise_distance': ['mean', 'std'],
        'max_pairwise_distance': ['mean', 'std'],
        'mean_centroid_distance': ['mean', 'std']
    }).round(4)
    
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(summary)
    
    summary_path = os.path.join(output_dir, "intra_sample_summary.csv")
    summary.to_csv(summary_path)
    print(f"\n✓ Summary saved to {summary_path}")
    
    return final_df, summary

if __name__ == "__main__":
    print("Calculating intra-sample distance metrics...")
    results_df, summary = compare_all_models()