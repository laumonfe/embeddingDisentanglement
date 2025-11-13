import pandas as pd
from itertools import combinations
import blue
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# Load dataset
csv_path = 'mamba_dataset/feidegger_mamba_metadata.csv'
df = pd.read_csv(csv_path)

# Group by item_idx (assuming column exists)
grouped = df.groupby('item_idx')['text'].apply(list)

smooth = SmoothingFunction().method1

results = []
all_bleu_scores = []  # Collect all BLEU scores globally
all_bleu_pairs = []   # Store (item_idx, idx_a, idx_b, bleu_ab, bleu_ba)

for item_idx, descriptions in tqdm(grouped.items(), desc="Processing items"):
    if len(descriptions) < 2:
        continue  # skip items with only one annotation

    bleu_scores = []
    for i, j in combinations(range(len(descriptions)), 2):
        a, b = descriptions[i], descriptions[j]
        bleu_ab = sentence_bleu([a.split()], b.split(), smoothing_function=smooth)
        bleu_ba = sentence_bleu([b.split()], a.split(), smoothing_function=smooth)
        bleu_scores.extend([bleu_ab, bleu_ba])
        all_bleu_scores.extend([bleu_ab, bleu_ba])
        all_bleu_pairs.append({
            'item_idx': item_idx,
            'desc_idx_a': i,
            'desc_idx_b': j,
            'bleu_ab': bleu_ab,
            'bleu_ba': bleu_ba
        })

    results.append({
        'item_idx': item_idx,
        'n_desc': len(descriptions),
        'bleu_mean': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
        'bleu_min': min(bleu_scores) if bleu_scores else 0,
        'bleu_max': max(bleu_scores) if bleu_scores else 0,
        'bleu_std': pd.Series(bleu_scores).std() if bleu_scores else 0,
    })

overview = pd.DataFrame(results)
print(overview.describe())
overview.to_csv('mamba_dataset/feidegger_intraclass_bleu_symetric.csv', index=False)
print('Saved overview to mamba_dataset/feidegger_intraclass_bleu_symetric.csv')

# Print global min and max BLEU scores
if all_bleu_scores:
    print(f"Global min BLEU: {min(all_bleu_scores)}")
    print(f"Global max BLEU: {max(all_bleu_scores)}")
    # Save all global BLEU scores and pairs
    bleu_pairs_df = pd.DataFrame(all_bleu_pairs)
    # Compute average BLEU for each pair
    bleu_pairs_df['bleu_avg'] = (bleu_pairs_df['bleu_ab'] + bleu_pairs_df['bleu_ba']) / 2
    bleu_pairs_df.to_csv('mamba_dataset/feidegger_global_bleu_pairs.csv', index=False)
    print('Saved all global BLEU pairs to mamba_dataset/feidegger_global_bleu_pairs.csv')
    print(bleu_pairs_df.describe())

    # Count how many pairs fall into each bin
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    bleu_pairs_df['bleu_bin'] = pd.cut(bleu_pairs_df['bleu_avg'], bins=bins, labels=labels, include_lowest=True, right=False)
    bin_counts = bleu_pairs_df['bleu_bin'].value_counts().sort_index()
    print("\nBLEU average score bin counts:")
    print(bin_counts)
else:
    print("No BLEU scores computed.")