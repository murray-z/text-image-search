# https://huggingface.co/docs/transformers/model_doc/clip


from transformers import AutoTokenizer, CLIPTextModel
from transformers import AutoProcessor, CLIPVisionModel
from config import CLIP_MODEL_NAME_OR_PATH
from PIL import Image
import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


device = get_device()


def load_text_model_tokenizer():
    text_model = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    text_tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    return text_model.to(device), text_tokenizer


def load_image_model_processor():
    img_model = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    img_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    return img_model.to(device), img_processor


def encoder_text(query, model, tokenizer):
    inputs = tokenizer([query], return_tensors="pt").to(device)
    outputs = model(**inputs)
    # last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output  # pooled (EOS token) states
    return pooled_output


def encoder_image(image_path, model, processor):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output  # pooled CLS states
    return pooled_output


