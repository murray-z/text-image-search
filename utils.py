from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from config import CLIP_MODEL_NAME_OR_PATH
from PIL import Image
import torch

CLIP_MODEL_NAME_OR_PATH = "./clip-vit-base-patch32"

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
    text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    text_tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    return text_model.to(device), text_tokenizer


def load_image_model_processor():
    img_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    img_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    return img_model.to(device), img_processor


def encoder_text(query, model, tokenizer):
    inputs = tokenizer([query], return_tensors="pt").to(device)
    outputs = model(**inputs)
    pooled_output = outputs.text_embeds  # pooled (EOS token) states
    if pooled_output is not None and pooled_output.shape[0] > 0:
        # print(pooled_output.shape)
        return list(pooled_output.cpu().detach().numpy()[0])
    else:
        return None


def encoder_image(image_path, model, processor):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    pooled_output = outputs.image_embeds  # pooled CLS states
    if pooled_output is not None and pooled_output.shape[0] > 0:
        # print(pooled_output.shape)
        return list(pooled_output.cpu().detach().numpy()[0])
    else:
        return None


if __name__ == '__main__':
    text_model, text_tokenizer = load_text_model_tokenizer()
    img_model, img_processor = load_image_model_processor()
    print(encoder_text("dog", text_model, text_tokenizer))
    print(encoder_image("./images/airport.jpg", img_model, img_processor))
