# Interactive Visualization of Sentence Embeddings with UMAP and Bokeh

import pandas as pd
import numpy as np
import umap
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, TapTool, Button, Div, LabelSet, TextInput
from bokeh.layouts import column, row
import os
from bokeh.io import curdoc


def get_split_embeddings(df, image_embeddings, text_embeddings, split_name):
    """
    Returns filtered DataFrame and corresponding image/text embeddings for a given split.
    Matches both 'idx' and 'desc_idx'.
    """
    if image_embeddings is None or text_embeddings is None:
        raise ValueError("Embeddings are None. Check that files were loaded correctly.")
    
    split_df = df[df["split"] == split_name]
    split_keys = set(zip(split_df["item_idx"], split_df["desc_idx"]))
    split_image_embeddings = [e for e in image_embeddings if (e['idx'], e['desc_idx']) in split_keys]
    split_text_embeddings = [e for e in text_embeddings if (e['idx'], e['desc_idx']) in split_keys]
    return split_df.reset_index(drop=True), np.array(split_image_embeddings, dtype=object), np.array(split_text_embeddings, dtype=object)


def load_embeddings(emb_save_path):
    if os.path.exists(emb_save_path):
        print(f"✓ Loading embeddings from {emb_save_path}")
        embeddings = np.load(emb_save_path, allow_pickle=True)
        print(f"  Contains: {len(embeddings)} embeddings.")
        return embeddings
    else:
        print(f"✗ ERROR: Embeddings file NOT FOUND")
        print(f"  Path: {emb_save_path}")
        print(f"  Absolute: {os.path.abspath(emb_save_path)}")
        print(f"  Current working directory: {os.getcwd()}")
        return None

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'item_idx' not in df.columns:
        df['item_idx'] = df.index
    return df

def create_id_selector(source):
    id_input = TextInput(title="Select item_idx:", placeholder="Enter item_idx...")
    id_input.js_on_change("value", CustomJS(args=dict(source=source), code="""
        const val = cb_obj.value;
        if (!val) {
            source.selected.indices = [];
            source.change.emit();
            return;
        }
        const indices = [];
        for (let i = 0; i < source.data['item_idx'].length; i++) {
            if (source.data['item_idx'][i].toString() === val) {
                indices.push(i);
            }
        }
        source.selected.indices = indices;
        source.change.emit();
    """))
    return id_input

def apply_umap(embeddings, n_neighbors=15, min_dist=0.1):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', random_state=42)
    umap_2d = umap_model.fit_transform(embeddings)
    return umap_2d

def create_interactive_plot(umap_2d, df):
    image_paths = df['image_path'].fillna('').tolist() if 'image_path' in df.columns else [''] * len(df)
    
    source_dict = {
        'x': umap_2d[:, 0],
        'y': umap_2d[:, 1],
        'text': df['text'].tolist(),
        'item_idx': df['item_idx'].tolist(),
        'image_path': image_paths,
        'label_num': [''] * len(df)
    }
    
    source = ColumnDataSource(data=source_dict)
    
    id_input = create_id_selector(source)
    
    hover = HoverTool(tooltips=[
        ("item_idx", "@item_idx"),
        ("text", "@text"),
        ("image_path", "@image_path"),
    ], mode='mouse') 
    
    image_div = Div(text="Click a point to see the image for the selected embedding.", width=400, height=340)
    texts_div = Div(text="<b>All texts for selected item_idx will appear here.</b>", width=400, height=340)
    texts_div.styles = {"overflow-y": "auto", "border": "1px solid #ccc", "padding": "8px"}

    callback = CustomJS(args=dict(source=source, image_div=image_div, texts_div=texts_div), code="""
        const indices = cb_data.source.selected.indices;
        if (indices.length === 0) {
            source.selected.indices = [];
            // Clear numbering
            for (let i = 0; i < source.data['label_num'].length; i++) {
                source.data['label_num'][i] = '';
            }
            source.change.emit();
            image_div.text = "Click a point to see the image for the selected embedding.";
            texts_div.text = "<b>All texts for selected item_idx will appear here.</b>";
            return;
        }
        const item_idx = source.data['item_idx'][indices[0]];
        const all_indices = [];
        let sentences = [];
        let last_image_path = "";
        let label_map = {};
        let label_counter = 1;
        for (let i = 0; i < source.data['item_idx'].length; i++) {
            if (source.data['item_idx'][i] === item_idx) {
                all_indices.push(i);
                label_map[i] = label_counter;
                sentences.push(label_counter + ". " + source.data['text'][i]);
                if (source.data['image_path'][i]) {
                    last_image_path = source.data['image_path'][i];
                }
                label_counter++;
            } else {
                source.data['label_num'][i] = '';
            }
        }
        // Assign numbers to selected points
        for (let idx of all_indices) {
            source.data['label_num'][idx] = label_map[idx].toString();
        }
        source.selected.indices = all_indices;
        source.change.emit();

        // Format sentences for display
        let texts_html = "<b>Sentences for item_idx " + item_idx + ":</b><ul>";
        for (let s of sentences) {
            texts_html += "<li>" + s + "</li>";
        }
        texts_html += "</ul>";
        texts_div.text = texts_html;

        // Show image for the group
        let image_html = "<b>Image for selected embedding group:</b><br><code>" + last_image_path + "</code>";
        if (last_image_path) {
            image_html += `<br><img src='${last_image_path}' width='300' style='margin-top:10px;'>`;
        } else {
            image_html += "<br><span style='color:red'>Image not found or could not be loaded.</span>";
        }
        image_div.text = image_html;
    """)

    taptool = TapTool(callback=callback)
    
    p = figure(
        title="UMAP Projection of Sentence Embeddings (Image on Select)",
        width=800, height=600,
        tools=["pan,wheel_zoom,reset,box_zoom,save", hover, taptool],
        output_backend='webgl'
    )
    
    renderer = p.scatter(
        'x', 'y', source=source, size=8, alpha=0.7, color="navy",
        selection_color="orange", selection_alpha=1.0, selection_line_color="red",
        nonselection_alpha=0.15, nonselection_color="gray",hover_color="lime", hover_alpha=1.0
    )
    
    labels = LabelSet(x='x', y='y', text='label_num', source=source,
                      text_font_size='12px', text_color='red',
                      x_offset=5, y_offset=5)
    p.add_layout(labels)
    
    clear_btn = Button(label="Clear Selection", button_type="default", width=150)
    clear_btn.js_on_click(CustomJS(args=dict(source=source, image_div=image_div, texts_div=texts_div), code="""
        source.selected.indices = [];
        for (let i = 0; i < source.data['label_num'].length; i++) {
            source.data['label_num'][i] = '';
        }
        source.change.emit();
        image_div.text = "Click a point to see the image for the selected embedding.";
        texts_div.text = "<b>All texts for selected item_idx will appear here.</b>";
    """))
    
    layout = column(id_input, p, clear_btn, row(image_div, texts_div))
    return layout


print(f"Current working directory: {os.getcwd()}")

# Paths relative to visualizations/ directory (where bokeh serve is run from)
csv_path = os.path.join("visualization_explorer", "static", "test_metadata.csv")
df = load_data(csv_path)

model_kind = "disentangled"  # "pretrained" or "finetuned", "disentangled"
data_type = "default" # "default" or "grouped"

# Go up one level (..) from visualizations/ to reach repo root, then into data/
emb_dir = os.path.join("..", "data", "embeddings", f"{model_kind}_{data_type}_clip-ViT-B-32-multilingual-v1")

img_emb_path_all = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}_{data_type}.npy")
text_emb_path_all = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}_{data_type}.npy")

print(f"\nLoading embeddings:")
image_embeddings = load_embeddings(img_emb_path_all)
text_embeddings = load_embeddings(text_emb_path_all)

if image_embeddings is None or text_embeddings is None:
    raise FileNotFoundError("Could not load embeddings. Check paths above.")

# Get test split
test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")
test_txt_emb_vectors = np.stack([e['embedding'] for e in test_txt_emb])

umap_2d = apply_umap(test_txt_emb_vectors)
layout = create_interactive_plot(umap_2d, test_df)
curdoc().add_root(layout)