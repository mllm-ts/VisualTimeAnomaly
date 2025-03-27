import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import base64
from io import BytesIO
import requests
import copy

def load_llava(model_name, device):
    if '8b' in model_name:
        device = 'cuda:7'
    elif '72b' in model_name:
        device = 'auto'

    # Initialize model and processor
    model = LlavaNextForConditionalGeneration.from_pretrained(
        f"llava-hf/{model_name}-hf",
        torch_dtype=torch.float16,
        load_in_4bit=True, 
        low_cpu_mem_usage=True, 
        device_map=device
    )

    return model

def call_llava(model_name, model, llava_request, device):
    processor = LlavaNextProcessor.from_pretrained(f"llava-hf/{model_name}-hf")
    # Process the conversation
    prompt = processor.apply_chat_template(llava_request['messages'], add_generation_prompt=True)
    image = llava_request['messages'][0]['content'][1]['image'],

    # Create model inputs
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    input_length = inputs.input_ids.shape[1]
    
    # Generate output
    outputs = model.generate(
        **inputs,
        **llava_request['config']
    )

    if '8b' in model_name:
        new_tokens = outputs[0][input_length+1:]
    elif '72b' in model_name:
        new_tokens = outputs[0][input_length:]

    # Decode and print result
    response = processor.decode(new_tokens, skip_special_tokens=True)

    return response


def convert_openai_to_llava(openai_request):    
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

    llava_messages = openai_request_copy["messages"]

    llava_config = {
        'temperature': openai_request.get("temperature", 0.4),
        'max_new_tokens': openai_request.get("max_tokens", 8192),
        'top_p': openai_request.get("top_p", 1.0),
        'do_sample': True
    }

    llava_request = {
        "messages": llava_messages,
        "config": llava_config
    }

    return llava_request

