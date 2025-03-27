import numpy as np
from scipy import interpolate
import json
import re
from scipy import stats

PROMPT_POINT = """Detect points of anomalies in this time series, in terms of the x-axis coordinate.
List one by one in a list. For example, if points x=2, 51, and 106 are anomalies, then output "[2, 51, 106]". If there are no anomalies, answer with an empty list [].
"""

PROMPT = """Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
List one by one in a list. For example, if ranges (incluing two endpoints) [2, 11], [50, 60], and [105, 118] are anomalies, then output "[[2, 11], [50, 60], [105, 118]]". \
If there are no anomalies, answer with an empty list [].
"""

PROMPT_VARIATE = """Detect univariate time series of anomalies in this multivariate time series, in terms of ID of univariate time series.
The image is a multivariate time series including multiple subimages to indicate multiple univariate time series. \
From left to right and top to bottom, the ID of each subimage increases by 1, starting from 0.
List one by one in a list. For example, if ID=0, 2, and 5 are anomalous univariate time series, then output "[0, 2, 5]". If there are no anomalies, answer with an empty list [].
"""

def encode_img(fig_path):
    import base64

    with open(fig_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_vision_messages(
    # time_series, 
    few_shots=False,
    cot=False,
    calc=None,
    # image_args={},
    data_tuple = None
):
    category, scenario, tsname, dim, drop_ratio, data_name, eval_i = data_tuple
    if category == 'synthetic':
        if scenario == 'univariate':
            fig_path = f'data/{category}/{scenario}/{data_name}/eval/fig/{eval_i:03d}.png'
        elif scenario == 'multivariate':
            fig_path = f'data/{category}/{scenario}/dim_{dim}/{data_name}/eval/fig/{eval_i:03d}.png'
        elif scenario.startswith('irr'):
            fig_path = f'data/{category}/{scenario}/ratio_{int(drop_ratio*100)}/{data_name}/eval/fig/{eval_i:03d}.png'
    elif category == 'semi':
        if scenario == 'univariate':
            fig_path = f'data/{category}/{scenario}/{tsname}/{data_name}/eval/fig/{eval_i:03d}.png'
        elif scenario == 'multivariate':
            fig_path = f'data/{category}/{scenario}/{tsname}/dim_{dim}/{data_name}/eval/fig/{eval_i:03d}.png'
        elif scenario.startswith('irr'):
            fig_path = f'data/{category}/{scenario}/{tsname}/ratio_{int(drop_ratio*100)}/{data_name}/eval/fig/{eval_i:03d}.png'

    img = encode_img(fig_path)

    if data_name in ["global", "contextual"]:
        prompt = PROMPT_POINT
    elif data_name in ["triangle", "square", "sawtooth", "random_walk"]:
        prompt = PROMPT_VARIATE
    else:
        prompt = PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"}
                },
            ],
        }
    ]
     
    return messages

def create_openai_request(
    # time_series,
    few_shots=False, 
    vision=False,
    temperature=0.4,
    stop=["’’’’", " – –", "<|endoftext|>", "<|eot_id|>"],
    cot=False,       # Chain of Thought
    calc=None,       # Enforce wrong calculation
    series_args={},  # Arguments for time_series_to_str
    # image_args={},   # Arguments for time_series_to_image
    data_tuple = None
):
    if vision:
        messages = create_vision_messages(few_shots, cot, calc, data_tuple)
    
    return {
        "messages": messages,
        "temperature": temperature,
        "stop": stop
    }
