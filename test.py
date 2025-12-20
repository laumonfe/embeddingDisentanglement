import os

from matplotlib import text

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sentence_transformers import  util
from src.models import PretrainedCLIPVision, PretrainedDistilBert, ProjectedCLIPVision, ProjectedDistilBert


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



text_encoder_baseline = PretrainedDistilBert("pretrained_models/sentence-transformers--clip-ViT-B-32-multilingual-v1", device)
model_kind = "disentangled"
dataset_type = "default"
txt_model_path_dis_def = f"output/{model_kind}_{dataset_type}_clip/epoch_20/text_encoder"
text_encoder_dis_def = ProjectedDistilBert(txt_model_path_dis_def, device)


query1 ="schwarzes kurzes Klied mit weißen blumen als Aufdruck es hat kurze Ärmel uund eine n V ausschnizt"
query2 = "Ein locker geschnittenes Mini-Wickelkleid in schwarz mit einem Blumenmuster in weiß, grau und pink. Das Kleid hat kurze Ärmel und einen tiefen V-Ausschnitt."

text_emb1 = text_encoder_baseline.encode(query1)
text_emb2 = text_encoder_baseline.encode(query2)
text_emb_dis1 = text_encoder_dis_def.encode(query1)
text_emb_dis2 = text_encoder_dis_def.encode(query2)
# Stack all embeddings into a matrix for similarity computation
sims_baseline = util.cos_sim(torch.tensor(text_emb1), torch.tensor(text_emb2))  
print(f"Baseline model similarity between '{query1}' and '{query2}': {sims_baseline.item():.4f}") 
sims_disentangled = util.cos_sim(torch.tensor(text_emb_dis1), torch.tensor(text_emb_dis2))  
print(f"Disentangled model similarity between '{query1}' and '{query2}': {sims_disentangled.item():.4f}")