from numpy import clip
from sentence_transformers import SentenceTransformer
import os

model_name = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
output_dir = "output/clip_model_files"
model = SentenceTransformer(model_name)
model.save(output_dir)

# print("Model files in:", output_dir)
# for fname in ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'sentencepiece.model', 'modules.json']:
#     fpath = os.path.join(output_dir, fname)
#     if os.path.exists(fpath):
#         print(f"{fname}: {fpath}")
#     else:
#         print(f"{fname}: NOT FOUND")

# print("\nAll files in model directory:")
# for f in os.listdir(output_dir):
#     print(f)

# print(model)

print("##########CLIP#####")
clip_model = "clip-ViT-B-32"
VIT_model = SentenceTransformer(clip_model)
print(VIT_model)

for i, module in enumerate(model._modules.values()):
    print(f"Module {i}: {module}")

clip_module = model._first_module()
print(clip_module.model.vision_model) 