import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import base64
from io import BytesIO
import requests
import copy

def load_qwen(model_name, device):
    if '7B' in model_name:
        device = 'cuda:7'
    elif '72B' in model_name:
        device = 'auto'

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        f"Qwen/{model_name}",
        torch_dtype=torch.float16,
        device_map=device
    )

    return model


def call_qwen(model_name, model, qwen_request, device):
    processor = AutoProcessor.from_pretrained(f"Qwen/{model_name}")
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    # Preparation for inference
    text = processor.apply_chat_template(
        qwen_request['messages'], tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(qwen_request['messages'])
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, **qwen_request['config'])
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return response[0]


def convert_openai_to_qwen(openai_request):
    openai_request_copy = copy.deepcopy(openai_request)

    for message in openai_request_copy["messages"]:
        for content in message["content"]:
            if content["type"] == "image_url":
                image_url = content["image_url"]["url"]
                if image_url.startswith("data:image"):
                    base64_str = image_url.split(",")[1]
                    img_data = base64.b64decode(base64_str)
                    img = Image.open(BytesIO(img_data))
                else:
                    response = requests.get(image_url)
                    img = Image.open(BytesIO(response.content))

                content.update({
                    "type": "image",
                    "image": img
                })
                content.pop("image_url")

    qwen_messages = openai_request_copy["messages"]

    qwen_config = {
        'temperature': openai_request.get("temperature", 0.4),
        'max_new_tokens': openai_request.get("max_tokens", 8192),
        'top_p': openai_request.get("top_p", 1.0),
        'do_sample': True
    }

    qwen_request = {
        "messages": qwen_messages,
        "config": qwen_config
    }

    return qwen_request