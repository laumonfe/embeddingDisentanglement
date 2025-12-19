# Interactive Visualization of Sentence Embeddings with UMAP and Bokeh

import pandas as pd
import numpy as np
import umap
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, TapTool, Button, Div, LabelSet, TextInput, Select
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


def load_embeddings(emb_save_path, status_callback=None):
    if status_callback:
        status_callback(f"📂 Loading file: {os.path.basename(emb_save_path)}...")
    
    if os.path.exists(emb_save_path):
        print(f"✓ Loading embeddings from {emb_save_path}")
        embeddings = np.load(emb_save_path, allow_pickle=True)
        print(f"  Contains: {len(embeddings)} embeddings.")
        if status_callback:
            status_callback(f"✓ Loaded {len(embeddings)} embeddings from {os.path.basename(emb_save_path)}")
        return embeddings
    else:
        print(f"✗ ERROR: Embeddings file NOT FOUND")
        print(f"  Path: {emb_save_path}")
        print(f"  Absolute: {os.path.abspath(emb_save_path)}")
        print(f"  Current working directory: {os.getcwd()}")
        if status_callback:
            status_callback(f"✗ File not found: {os.path.basename(emb_save_path)}")
        return None

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'item_idx' not in df.columns:
        df['item_idx'] = df.index
    return df

def load_and_update_plot(df, model_kind, data_type, embedding_type="text", status_callback=None):
    """Load new embeddings and update the plot."""
    # Handle different path structure for baseline model
    if model_kind == "baseline":
        emb_dir = os.path.join("..", "data", "embeddings", "baseline_clip-ViT-B-32-multilingual-v1")
        img_emb_path = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_baseline.npy")
        text_emb_path = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_baseline.npy")
    else:
        emb_dir = os.path.join("..", "data", "embeddings", f"{model_kind}_{data_type}_clip-ViT-B-32-multilingual-v1")
        img_emb_path = os.path.join(emb_dir, f"image_embeddings_clip-ViT-B-32_{model_kind}_{data_type}.npy")
        text_emb_path = os.path.join(emb_dir, f"text_embeddings_clip-ViT-B-32-multilingual-v1_{model_kind}_{data_type}.npy")
    
    print(f"\nLoading embeddings for {model_kind}/{data_type}/{embedding_type}:")
    if status_callback:
        status_callback(f"🔄 Loading embeddings for {model_kind}/{data_type}/{embedding_type}...")
    
    image_embeddings = load_embeddings(img_emb_path, status_callback)
    text_embeddings = load_embeddings(text_emb_path, status_callback)
    
    if image_embeddings is None or text_embeddings is None:
        print("Failed to load embeddings!")
        if status_callback:
            status_callback("✗ Failed to load embeddings!")
        return None, None
    
    if status_callback:
        status_callback(f"🔍 Filtering test split...")
    
    test_df, test_img_emb, test_txt_emb = get_split_embeddings(df, image_embeddings, text_embeddings, "test")
    
    if embedding_type == "text":
        test_emb_vectors = np.stack([e['embedding'] for e in test_txt_emb])
        metric = 'cosine'
    else:  # image
        test_emb_vectors = np.stack([e['embedding'] for e in test_img_emb])
        metric = 'euclidean'
    
    if status_callback:
        status_callback(f"🎨 Computing UMAP projection ({metric} metric)...")
    
    umap_2d = apply_umap(test_emb_vectors, metric=metric)
    
    if status_callback:
        status_callback(f"✓ Successfully computed UMAP for {len(test_df)} points")
    
    return umap_2d, test_df

def create_id_selector(source, image_div, texts_div):
    id_input = TextInput(title="Select item_idx:", placeholder="Enter item_idx...")
    id_input.js_on_change("value", CustomJS(args=dict(source=source, image_div=image_div, texts_div=texts_div), code="""
        const val = cb_obj.value.trim();
        if (!val) {
            source.selected.indices = [];
            for (let i = 0; i < source.data['label_num'].length; i++) {
                source.data['label_num'][i] = '';
            }
            source.change.emit();
            image_div.text = "Click a point to see the image for the selected embedding.";
            texts_div.text = "<b>All texts for selected item_idx will appear here.</b>";
            return;
        }
        const indices = [];
        let label_counter = 1;
        let sentences = [];
        let last_image_path = "";
        
        for (let i = 0; i < source.data['item_idx'].length; i++) {
            if (source.data['item_idx'][i].toString() === val) {
                indices.push(i);
                source.data['label_num'][i] = label_counter.toString();
                sentences.push(label_counter + ". " + source.data['text'][i]);
                if (source.data['image_path'][i]) {
                    last_image_path = source.data['image_path'][i];
                }
                label_counter++;
            } else {
                source.data['label_num'][i] = '';
            }
        }
        source.selected.indices = indices;
        source.change.emit();
        
        // Update texts div
        if (indices.length > 0) {
            let texts_html = "<b>Sentences for item_idx " + val + ":</b><ul>";
            for (let s of sentences) {
                texts_html += "<li>" + s + "</li>";
            }
            texts_html += "</ul>";
            texts_div.text = texts_html;
            
            // Update image div
            let image_html = "<b>Image for selected embedding group:</b><br><code>" + last_image_path + "</code>";
            if (last_image_path) {
                image_html += `<br><img src='${last_image_path}' width='300' style='margin-top:10px;'>`;
            } else {
                image_html += "<br><span style='color:red'>Image not found or could not be loaded.</span>";
            }
            image_div.text = image_html;
        } else {
            image_div.text = "No matching item_idx found.";
            texts_div.text = "<b>No matching item_idx found.</b>";
        }
    """))
    return id_input

def apply_umap(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine'):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    umap_2d = umap_model.fit_transform(embeddings)
    return umap_2d

def create_interactive_plot(umap_2d, df, initial_model, initial_data, initial_embedding):
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
    
    image_div = Div(text="Click a point to see the image for the selected embedding.", width=400, height=340)
    texts_div = Div(text="<b>All texts for selected item_idx will appear here.</b>", width=400, height=340)
    texts_div.styles = {"overflow-y": "auto", "border": "1px solid #ccc", "padding": "8px"}
    
    id_input = create_id_selector(source, image_div, texts_div)
    
    hover = HoverTool(tooltips=[
        ("item_idx", "@item_idx"),
        ("text", "@text"),
        ("image_path", "@image_path"),
    ], mode='mouse') 

    callback = CustomJS(args=dict(source=source, image_div=image_div, texts_div=texts_div, id_input=id_input), code="""
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
            id_input.value = "";
            return;
        }
        const item_idx = source.data['item_idx'][indices[0]];
        
        // Update the text input to show the selected item_idx
        id_input.value = item_idx.toString();
        
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
        title=f"UMAP Projection: {initial_model}/{initial_data}/{initial_embedding}",
        width=800, height=600,
        tools=["pan,wheel_zoom,reset,box_zoom,save", hover, taptool],
        output_backend='webgl'
    )
    
    renderer = p.scatter(
        'x', 'y', source=source, size=8, alpha=0.7, color="navy",
        selection_color="orange", selection_alpha=1.0, selection_line_color="red",
        nonselection_alpha=0.15, nonselection_color="gray", hover_color="lime", hover_alpha=1.0
    )
    
    labels = LabelSet(x='x', y='y', text='label_num', source=source,
                      text_font_size='12px', text_color='red',
                      x_offset=5, y_offset=5)
    p.add_layout(labels)
    
    clear_btn = Button(label="Clear Selection", button_type="default", width=150)
    clear_btn.js_on_click(CustomJS(args=dict(source=source, image_div=image_div, texts_div=texts_div, id_input=id_input), code="""
        source.selected.indices = [];
        for (let i = 0; i < source.data['label_num'].length; i++) {
            source.data['label_num'][i] = '';
        }
        source.change.emit();
        id_input.value = "";
        image_div.text = "Click a point to see the image for the selected embedding.";
        texts_div.text = "<b>All texts for selected item_idx will appear here.</b>";
    """))
    
    # Create dropdown controls
    model_select = Select(title="Model:", value=initial_model, 
                         options=["baseline", "finetuned", "disentangled"], width=150)
    data_select = Select(title="Data Type:", value=initial_data,
                        options=["default", "grouped"], width=150)
    embedding_select = Select(title="Embedding Type:", value=initial_embedding,
                             options=["text", "image"], width=150)
    
    update_btn = Button(label="Load Embeddings", button_type="success", width=150)
    status_div = Div(text=f"Currently showing: {initial_model}/{initial_data}/{initial_embedding}", width=800)
    status_div.styles = {"padding": "10px", "background-color": "#f0f0f0", "border": "1px solid #ccc", "border-radius": "5px"}
    
    # Store the original CSV dataframe globally for reloading
    csv_df = load_data(os.path.join("visualization_explorer", "static", "test_metadata.csv"))
    
    def update_status(message):
        """Helper function to update status div from callback"""
        status_div.text = f"<div style='font-size: 14px;'>{message}</div>"
    
    def async_load_embeddings(model_kind, data_type, embedding_type):
        """The actual loading function that runs asynchronously"""
        try:
            new_umap_2d, new_df = load_and_update_plot(csv_df, model_kind, data_type, embedding_type, 
                                                       status_callback=lambda msg: curdoc().add_next_tick_callback(lambda: update_status(msg)))
            
            if new_umap_2d is not None:
                def update_visualization():
                    update_status("📊 Updating plot...")
                    
                    # Update source data
                    new_image_paths = new_df['image_path'].fillna('').tolist() if 'image_path' in new_df.columns else [''] * len(new_df)
                    source.data = {
                        'x': new_umap_2d[:, 0],
                        'y': new_umap_2d[:, 1],
                        'text': new_df['text'].tolist(),
                        'item_idx': new_df['item_idx'].tolist(),
                        'image_path': new_image_paths,
                        'label_num': [''] * len(new_df)
                    }
                    
                    # Update title based on model type
                    if model_kind == "baseline":
                        p.title.text = f"UMAP Projection: {model_kind}/{embedding_type}"
                    else:
                        p.title.text = f"UMAP Projection: {model_kind}/{data_type}/{embedding_type}"
                    
                    update_status(f"✅ Successfully loaded {model_kind}/{data_type}/{embedding_type} ({len(new_df)} points)")
                    
                    # Clear selections and divs
                    id_input.value = ""
                    image_div.text = "Click a point to see the image for the selected embedding."
                    texts_div.text = "<b>All texts for selected item_idx will appear here.</b>"
                    
                    # Re-enable button
                    update_btn.disabled = False
                    update_btn.label = "Load Embeddings"
                
                curdoc().add_next_tick_callback(update_visualization)
            else:
                def show_error():
                    update_status("❌ Failed to load embeddings. Check console for details.")
                    update_btn.disabled = False
                    update_btn.label = "Load Embeddings"
                
                curdoc().add_next_tick_callback(show_error)
        except Exception as e:
            def show_exception():
                update_status(f"❌ Error: {str(e)}")
                update_btn.disabled = False
                update_btn.label = "Load Embeddings"
            
            print(f"Error loading embeddings: {e}")
            import traceback
            traceback.print_exc()
            curdoc().add_next_tick_callback(show_exception)
    
    def update_plot():
        model_kind = model_select.value
        data_type = data_select.value
        embedding_type = embedding_select.value
        
        # Disable data_type selector for baseline
        if model_kind == "baseline":
            data_type = "default"  # baseline doesn't use data_type
        
        # Disable button during loading
        update_btn.disabled = True
        update_btn.label = "Loading..."
        update_status(f"🔄 Starting to load {model_kind}/{data_type}/{embedding_type}...")
        
        # Schedule the async loading
        curdoc().add_next_tick_callback(lambda: async_load_embeddings(model_kind, data_type, embedding_type))
    
    update_btn.on_click(update_plot)
    
    controls = row(model_select, data_select, embedding_select, update_btn)
    layout = column(controls, status_div, id_input, p, clear_btn, row(image_div, texts_div))
    return layout


print(f"Current working directory: {os.getcwd()}")

# Paths relative to visualizations/ directory (where bokeh serve is run from)
csv_path = os.path.join("visualization_explorer", "static", "test_metadata.csv")
df = load_data(csv_path)

# Initial settings
model_kind = "disentangled"  # "baseline", "finetuned", or "disentangled"
data_type = "default"  # "default" or "grouped"
embedding_type = "text"  # "text" or "image"

print(f"\nInitial load: {model_kind}/{data_type}/{embedding_type}")
print("🔄 Loading initial embeddings...")

def initial_status_print(message):
    print(message)

umap_2d, test_df = load_and_update_plot(df, model_kind, data_type, embedding_type, 
                                        status_callback=initial_status_print)

if umap_2d is not None:
    print("✅ Initial embeddings loaded successfully!")
    layout = create_interactive_plot(umap_2d, test_df, model_kind, data_type, embedding_type)
    curdoc().add_root(layout)
else:
    print("❌ Failed to create initial plot!")
    error_div = Div(text="<h2>Error: Failed to load initial embeddings. Check console.</h2>")
    curdoc().add_root(error_div)