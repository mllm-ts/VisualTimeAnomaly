import google.generativeai as genai
from PIL import Image
from loguru import logger
import yaml
import requests
from io import BytesIO
import base64

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def load_gemini(model_name):
    credentials = yaml.safe_load(open("credentials.yml"))
    assert model_name in credentials, f"Model {model_name} not found in credentials"

    credential = credentials[model_name]
    api_key = credential["api_key"]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    return model

def call_gemini(model_name, model, gemini_request):    
    logger.debug(f"{model_name} is running")
    
    response = model.generate_content(
        contents=gemini_request['messages'],
        generation_config=gemini_request['config'],
        # **gemini_request,
        safety_settings=SAFETY_SETTINGS,
    )
    return response.text


def convert_openai_to_gemini(openai_request):
    gemini_messages = []

    for message in openai_request["messages"]:
        parts = []
        for content in message["content"]:
            if content["type"] == "text":
                parts.append(content["text"])
            elif content["type"] == "image_url":
                image_url = content["image_url"]["url"]
                if image_url.startswith("data:image"):
                    base64_str = image_url.split(",")[1]
                    img_data = base64.b64decode(base64_str)
                    img = Image.open(BytesIO(img_data))
                else:
                    response = requests.get(image_url)
                    img = Image.open(BytesIO(response.content))
                parts.append(img)
        
        gemini_messages.append({"role": message["role"].replace("assistant", "model"), "parts": parts})

    gemini_config = {
        'temperature': openai_request.get("temperature", 0.4),
        'max_output_tokens': openai_request.get("max_tokens", 8192),
        'top_p': openai_request.get("top_p", 1.0),
        "stop_sequences": openai_request.get("stop", [])
    }

    gemini_request = {
        "messages": gemini_messages,
        "config": gemini_config
    }

    return gemini_request
