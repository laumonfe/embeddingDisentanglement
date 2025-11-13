import pandas as pd
from itertools import combinations
from rouge_score import rouge_scorer
from tqdm import tqdm

# Load dataset
csv_path = 'mamba_dataset/feidegger_mamba_metadata.csv'
df = pd.read_csv(csv_path)

# Group by item_idx (assuming column exists)
grouped = df.groupby('item_idx')['text'].apply(list)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

results = []
all_rougeL_scores = []
all_rougeL_pairs = []

for item_idx, descriptions in tqdm(grouped.items(), desc="Processing items"):
    if len(descriptions) < 2:
        continue  # skip items with only one annotation

    rougeL_scores = []
    for i, j in combinations(range(len(descriptions)), 2):
        a, b = descriptions[i], descriptions[j]
        rougeL_ab = scorer.score(a, b)['rougeL'].fmeasure
        rougeL_ba = scorer.score(b, a)['rougeL'].fmeasure
        rougeL_scores.extend([rougeL_ab, rougeL_ba])
        all_rougeL_scores.extend([rougeL_ab, rougeL_ba])
        all_rougeL_pairs.append({
            'item_idx': item_idx,
            'desc_idx_a': i,
            'desc_idx_b': j,
            'rougeL_ab': rougeL_ab,
            'rougeL_ba': rougeL_ba
        })

    results.append({
        'item_idx': item_idx,
        'n_desc': len(descriptions),
        'rougeL_mean': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0,
        'rougeL_min': min(rougeL_scores) if rougeL_scores else 0,
        'rougeL_max': max(rougeL_scores) if rougeL_scores else 0,
        'rougeL_std': pd.Series(rougeL_scores).std() if rougeL_scores else 0,
    })

overview = pd.DataFrame(results)
print(overview.describe())
overview.to_csv('mamba_dataset/feidegger_intraclass_rougeL_symetric.csv', index=False)
print('Saved overview to mamba_dataset/feidegger_intraclass_rougeL_symetric.csv')

# Print global min and max ROUGE-L scores
if all_rougeL_scores:
    print(f"Global min ROUGE-L: {min(all_rougeL_scores)}")
    print(f"Global max ROUGE-L: {max(all_rougeL_scores)}")
    # Save all global ROUGE-L scores and pairs
    rougeL_pairs_df = pd.DataFrame(all_rougeL_pairs)
    # Compute average ROUGE-L for each pair
    rougeL_pairs_df['rougeL_avg'] = (rougeL_pairs_df['rougeL_ab'] + rougeL_pairs_df['rougeL_ba']) / 2
    rougeL_pairs_df.to_csv('mamba_dataset/feidegger_global_rougeL_pairs.csv', index=False)
    print('Saved all global ROUGE-L pairs to mamba_dataset/feidegger_global_rougeL_pairs.csv')
    print(rougeL_pairs_df.describe())

    # Count how many pairs fall into each bin
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    rougeL_pairs_df['rougeL_bin'] = pd.cut(rougeL_pairs_df['rougeL_avg'], bins=bins, labels=labels, include_lowest=True, right=False)
    bin_counts = rougeL_pairs_df['rougeL_bin'].value_counts().sort_index()
    print("\nROUGE-L average score bin counts:")
    print(bin_counts)
else:
    print("No ROUGE-L scores computed.")