import transformers
import torch
from transformers import pipeline

pipe = pipeline(model="padmajabfrl/Gender-Classification")
pipe.model.config.id2label
