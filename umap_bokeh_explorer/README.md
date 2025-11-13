# Interactive Visualization of Sentence Embeddings with UMAP and Bokeh

This project provides an interactive visualization of sentence embeddings using UMAP for dimensionality reduction and Bokeh for visualization. It allows users to explore high-dimensional sentence embeddings in a 2D space, with the option to display associated images upon selection.

## Project Structure

```
umap_bokeh_explorer/
├── src/
│   ├── umap_bokeh_explorer.py  # Main script for visualization
│   └── utils.py                 # Utility functions
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies. You can do this using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your dataset**: Ensure your dataset is in CSV format with a column for text data and optionally a column for image paths.

2. **Run the main script**: Execute the main script to start the interactive visualization.

```bash
bokeh serve --show umap_bokeh_explorer
```

3. **Interact with the visualization**: Click on points in the scatter plot to view the corresponding sentence and image.

