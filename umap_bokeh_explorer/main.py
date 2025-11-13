# Interactive Visualization of Sentence Embeddings with UMAP and Bokeh

import pandas as pd
import numpy as np
import umap
from sentence_transformers import SentenceTransformer
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, TapTool, Button, Div, LabelSet
from bokeh.layouts import column, row
import os
from bokeh.io import curdoc

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'item_idx' not in df.columns:
        df['item_idx'] = df.index
    return df

def compute_embeddings(df, embeddings_path='data/feidegger_embeddings.npy', model_name='distiluse-base-multilingual-cased-v2'):
    if os.path.exists(embeddings_path):
        print(f"Loading embeddings from {embeddings_path} ...")
        embeddings = np.load(embeddings_path)
    else:
        print("Calculating embeddings ...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path}")
    return embeddings

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
    
    hover = HoverTool(tooltips=[
        ("Sentence", "@text"),
        ("Item idx", "@item_idx"),
        ("Image path", "@image_path"),
        ("Number", "@label_num"),
    ])
    
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
        nonselection_alpha=0.15, nonselection_color="gray"
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
    
    layout = column(p, clear_btn, row(image_div, texts_div))
    return layout

csv_path = 'mamba_dataset/feidegger_mamba_metadata_vis.csv'
df = load_data(csv_path)
embeddings = compute_embeddings(df)
umap_2d = apply_umap(embeddings)
layout = create_interactive_plot(umap_2d, df)
curdoc().add_root(layout)