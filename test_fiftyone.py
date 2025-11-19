import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F

dataset_name = "clustering-demo2"

# Check if dataset exists, load if so, otherwise create from zoo
if dataset_name in fo.list_datasets():
    dataset = fo.load_dataset(dataset_name)
else:
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        dataset_name=dataset_name,
        persistent=True
    )
    # delete labels to simulate starting with unlabeled data
    dataset.select_fields().keep_fields()
    dataset.persistent = True
# launch the app to visualize the dataset
session = fo.launch_app(dataset)

res = fob.compute_visualization(
    dataset,
    model="clip-vit-base32-torch",
    embeddings="clip_embeddings",
    method="umap",
    brain_key="clip_vis",
    batch_size=10
)
dataset.set_values("clip_umap", res.current_points)
session.wait()