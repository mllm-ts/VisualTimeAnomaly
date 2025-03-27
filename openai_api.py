from loguru import logger
from openai import AzureOpenAI
import yaml

def load_gpt(model_name):
    credentials = yaml.safe_load(open("credentials.yml"))
    assert model_name in credentials, f"Model {model_name} not found in credentials"

    credential = credentials[model_name]
    api_key = credential["api_key"]
    api_version = credential["api_version"]
    base_url = credential["base_url"]

    model = AzureOpenAI(
        api_key=api_key,  
        api_version=api_version,
        base_url=base_url
    )

    return model

def call_gpt(model_name, model, openai_request):
    logger.debug(f"{model_name} is running")

    response = model.chat.completions.create(
        model=model_name, **openai_request
    )
    return response.choices[0].message.content

