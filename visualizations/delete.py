import fiftyone as fo
# List all datasets
datasets = fo.list_datasets()
print(f"Found {len(datasets)} datasets: {datasets}")

# Delete all datasets
for dataset_name in datasets:
    fo.delete_dataset(dataset_name)
    print(f"Deleted: {dataset_name}")

print("All datasets deleted!")