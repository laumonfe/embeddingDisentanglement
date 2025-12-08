# Visualizations Overview

This folder contains scripts and tools for visualizing multimodal embeddings and retrieval results using the FEIDEGGER dataset.

## Available Visualization Methods

### 1. **FiftyOne App (`fiftyone_app.py`)**
- **Purpose:**  
  Interactive exploration of image and text embeddings using the FiftyOne platform.
- **Features:**  
  - Loads image and text embeddings for a given split (e.g., test set).
  - Creates a FiftyOne dataset with samples containing images, text, and their embeddings.
  - Computes UMAP projections for both text and image embeddings, enabling visual inspection of clustering and structure.
  - Launches the FiftyOne web app for interactive browsing, filtering, and similarity search.

<video width="100%" controls>
  <source src="../assets/fiftyOne.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### 2. **Retrieval Visualization (`retrieval_visualization.py`)**
- **Purpose:**  
  Visualize retrieval results, such as top-k matches for a given query.
- **Features:**  
  - Displays images retrieved for a text query.
  - Displays iamges retrieved for the top image taht was retrieved by the text query. 
  - Useful for evaluating model performance and inspecting qualitative results.

<img src="../assets/text2img.png" alt="Example Retrieval Result" style="width:100%;">
<img src="../assets/img2img.png" alt="Example Retrieval Result" style="width:100%;">

### 3. **Visualization Explorer**
- **Purpose:**  
  Visualize clusters in the embedding space.
- **Features:**  
  - Uses dimensionality reduction techniques (UMAP, t-SNE) to project high-dimensional embeddings.
  - Shows how images or texts group together, revealing semantic or stylistic patterns.

<video width="100%" controls>
  <source src="../assets/umap_visualization.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>



**Note:**  
Make sure you have installed all dependencies listed in `requirements.txt` before running visualization scripts.