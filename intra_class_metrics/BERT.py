import pandas as pd
from itertools import combinations
from tqdm import tqdm
import bert_score
import torch
from transformers import AutoModel, AutoTokenizer


# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load dataset
csv_path = 'mamba_dataset/feidegger_mamba_metadata.csv'
df = pd.read_csv(csv_path)
df = df[:100]

# Group by item_idx (assuming column exists)
grouped = df.groupby('item_idx')['text'].apply(list)



model_type = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_type)
model = AutoModel.from_pretrained(model_type)

results = []
all_bertscore_scores = []
all_bertscore_pairs = []

for idx, (item_idx, descriptions) in enumerate(tqdm(grouped.items(), desc="Processing items")):
    if idx % 100 == 0 and idx > 0:
        print(f"Processed {idx} items out of {len(grouped)}...")
    if len(descriptions) < 2:
        continue  # skip items with only one annotation

    # Prepare all unique pairs
    pairs = list(combinations(range(len(descriptions)), 2))
    if not pairs:
        continue

    # Prepare candidate and reference lists for both directions
    cands_ab = [descriptions[i] for i, j in pairs]
    refs_ab = [descriptions[j] for i, j in pairs]
    cands_ba = [descriptions[j] for i, j in pairs]
    refs_ba = [descriptions[i] for i, j in pairs]
    batch_size = 2048
    # Compute BERTScore F1 for all pairs in both directions (batched)
    P_ab, R_ab, F1_ab = bert_score.score(
        cands_ab, refs_ab, lang='de', rescale_with_baseline=False, device=device,
        model_type=model_type, batch_size=batch_size
    )
    P_ba, R_ba, F1_ba = bert_score.score(
        cands_ba, refs_ba, lang='de', rescale_with_baseline=False, device=device,
        model_type=model_type, batch_size=batch_size
    )

    bertscore_scores = []
    for idx_pair, ((i, j), f1_ab, f1_ba) in enumerate(zip(pairs, F1_ab, F1_ba)):
        avg_f1 = (f1_ab.item() + f1_ba.item()) / 2
        bertscore_scores.extend([f1_ab.item(), f1_ba.item()])
        all_bertscore_scores.extend([f1_ab.item(), f1_ba.item()])
        all_bertscore_pairs.append({
            'item_idx': item_idx,
            'desc_idx_a': i,
            'desc_idx_b': j,
            'bertscore_ab': f1_ab.item(),
            'bertscore_ba': f1_ba.item(),
            'bertscore_avg': avg_f1
        })

    results.append({
        'item_idx': item_idx,
        'n_desc': len(descriptions),
        'bertscore_mean': sum(bertscore_scores) / len(bertscore_scores) if bertscore_scores else 0,
        'bertscore_min': min(bertscore_scores) if bertscore_scores else 0,
        'bertscore_max': max(bertscore_scores) if bertscore_scores else 0,
        'bertscore_std': pd.Series(bertscore_scores).std() if bertscore_scores else 0,
    })

overview = pd.DataFrame(results)
print(overview.describe())
overview.to_csv('mamba_dataset/feidegger_intraclass_bertscore_symetric.csv', index=False)
print('Saved overview to mamba_dataset/feidegger_intraclass_bertscore_symetric.csv')

# Print global min and max BERTScore F1 scores
if all_bertscore_scores:
    print(f"Global min BERTScore F1: {min(all_bertscore_scores)}")
    print(f"Global max BERTScore F1: {max(all_bertscore_scores)}")
    # Save all global BERTScore F1 scores and pairs
    bertscore_pairs_df = pd.DataFrame(all_bertscore_pairs)
    bertscore_pairs_df.to_csv('mamba_dataset/feidegger_global_bertscore_pairs.csv', index=False)
    print('Saved all global BERTScore pairs to mamba_dataset/feidegger_global_bertscore_pairs.csv')
    print(bertscore_pairs_df.describe())

    # Count how many pairs fall into each bin
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    bertscore_pairs_df['bertscore_bin'] = pd.cut(bertscore_pairs_df['bertscore_avg'], bins=bins, labels=labels, include_lowest=True, right=False)
    bin_counts = bertscore_pairs_df['bertscore_bin'].value_counts().sort_index()
    print("\nBERTScore average F1 bin counts:")
    print(bin_counts)
else:
    print("No BERTScore F1 scores computed.")