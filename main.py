from openai_api import load_gpt, call_gpt
from gemini_api import load_gemini, call_gemini, convert_openai_to_gemini
from llava_api import load_llava, call_llava, convert_openai_to_llava
from qwen_api import load_qwen, call_qwen, convert_openai_to_qwen
from config import create_api_configs
from utils import process_request
import argparse
from dataloader import TSIDataset

def load_mllm(model_name, device):
    if model_name.startswith('gpt'):
        return load_gpt(model_name)
    elif model_name.startswith('gemini'):
        return load_gemini(model_name)
    elif 'llava' in model_name:
        return load_llava(model_name, device)
    elif model_name.startswith('Qwen'):
        return load_qwen(model_name, device)

def call_mllm(model_name, model, request, device):
    if model_name.startswith('gpt'):
        response = call_gpt(model_name, model, request)
    elif model_name.startswith('gemini'):
        response = call_gemini(model_name, model, convert_openai_to_gemini(request))
    elif 'llava' in model_name:
        response = call_llava(model_name, model, convert_openai_to_llava(request), device)
    elif model_name.startswith('Qwen'):
        response = call_qwen(model_name, model, convert_openai_to_qwen(request), device)
    
    return response

# The code is adapted from https://github.com/rose-stl-lab/anomllm
def AD_with_retries(
    model_name: str,
    category: str,
    scenario: str,
    tsname: str,
    data_name: str,
    request_func: callable,
    variant: str = "standard",
    num_retries: int = 4,
    dim: int = 9,
    drop_ratio: float = 0.00,
    device: str = 'cuda:7'
):
    import json
    import time
    import pickle
    import os
    from loguru import logger

    results = {}

    if category == 'synthetic':
        if scenario == 'univariate':
            log_fn = f"logs/{category}/{scenario}/{data_name}/{model_name}/" + variant + ".log"
            logger.add(log_fn, format="{time} {level} {message}", level="INFO")
            results_dir = f'results/{category}/{scenario}/{data_name}/{model_name}'
            data_dir = f'data/{category}/{scenario}/{data_name}/eval'
            train_dir = f'data/{category}/{scenario}/{data_name}/train'
            jsonl_fn = os.path.join(results_dir, variant + '.jsonl')
        elif scenario == 'multivariate':
            log_fn = f"logs/{category}/{scenario}/dim_{dim}/{data_name}/{model_name}/" + variant + ".log"
            logger.add(log_fn, format="{time} {level} {message}", level="INFO")
            results_dir = f'results/{category}/{scenario}/dim_{dim}/{data_name}/{model_name}'
            data_dir = f'data/{category}/{scenario}/dim_{dim}/{data_name}/eval'
            train_dir = f'data/{category}/{scenario}/dim_{dim}/{data_name}/train'
            jsonl_fn = os.path.join(results_dir, variant + '.jsonl')
        elif scenario.startswith('irr'):
            log_fn = f"logs/{category}/{scenario}/ratio_{int(drop_ratio*100)}/{data_name}/{model_name}/" + variant + ".log"
            logger.add(log_fn, format="{time} {level} {message}", level="INFO")
            results_dir = f'results/{category}/{scenario}/ratio_{int(drop_ratio*100)}/{data_name}/{model_name}'
            data_dir = f'data/{category}/{scenario}/ratio_{int(drop_ratio*100)}/{data_name}/eval'
            train_dir = f'data/{category}/{scenario}/ratio_{int(drop_ratio*100)}/{data_name}/train'
            jsonl_fn = os.path.join(results_dir, variant + '.jsonl')
    elif category == 'semi':
        if scenario == 'univariate':
            log_fn = f"logs/{category}/{scenario}/{tsname}/{data_name}/{model_name}/" + variant + ".log"
            logger.add(log_fn, format="{time} {level} {message}", level="INFO")
            results_dir = f'results/{category}/{scenario}/{tsname}/{data_name}/{model_name}'
            data_dir = f'data/{category}/{scenario}/{tsname}/{data_name}/eval'
            train_dir = f'data/{category}/{scenario}/{tsname}/{data_name}/train'
            jsonl_fn = os.path.join(results_dir, variant + '.jsonl')
        elif scenario == 'multivariate':
            log_fn = f"logs/{category}/{scenario}/{tsname}/dim_{dim}/{data_name}/{model_name}/" + variant + ".log"
            logger.add(log_fn, format="{time} {level} {message}", level="INFO")
            results_dir = f'results/{category}/{scenario}/{tsname}/dim_{dim}/{data_name}/{model_name}'
            data_dir = f'data/{category}/{scenario}/{tsname}/dim_{dim}/{data_name}/eval'
            train_dir = f'data/{category}/{scenario}/{tsname}/dim_{dim}/{data_name}/train'
            jsonl_fn = os.path.join(results_dir, variant + '.jsonl')
        elif scenario.startswith('irr'):
            log_fn = f"logs/{category}/{scenario}/{tsname}/ratio_{int(drop_ratio*100)}/{data_name}/{model_name}/" + variant + ".log"
            logger.add(log_fn, format="{time} {level} {message}", level="INFO")
            results_dir = f'results/{category}/{scenario}/{tsname}/ratio_{int(drop_ratio*100)}/{data_name}/{model_name}'
            data_dir = f'data/{category}/{scenario}/{tsname}/ratio_{int(drop_ratio*100)}/{data_name}/eval'
            train_dir = f'data/{category}/{scenario}/{tsname}/ratio_{int(drop_ratio*100)}/{data_name}/train'
            jsonl_fn = os.path.join(results_dir, variant + '.jsonl')

    os.makedirs(results_dir, exist_ok=True)
    
    eval_dataset = TSIDataset(data_dir)
    train_dataset = TSIDataset(train_dir)

    # Load existing results if jsonl file exists
    if os.path.exists(jsonl_fn):
        with open(jsonl_fn, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                results[entry['custom_id']] = entry["response"]

    # Loop over image files
    model = load_mllm(model_name, device)
    for i in range(1, len(eval_dataset) + 1):
        if category == 'synthetic':
            if scenario == 'univariate':
                custom_id = f"{category}_{scenario}_{data_name}_{model_name}_{variant}_{str(i).zfill(5)}"
            elif scenario == 'multivariate':
                custom_id = f"{category}_{scenario}_dim_{dim}_{data_name}_{model_name}_{variant}_{str(i).zfill(5)}"
            elif scenario.startswith('irr'):
                custom_id = f"{category}_{scenario}_ratio_{int(drop_ratio*100)}_{data_name}_{model_name}_{variant}_{str(i).zfill(5)}"
        elif category == 'semi':
            if scenario == 'univariate':
                custom_id = f"{category}_{scenario}_{tsname}_{data_name}_{model_name}_{variant}_{str(i).zfill(5)}"
            elif scenario == 'multivariate':
                custom_id = f"{category}_{scenario}_{tsname}_dim_{dim}_{data_name}_{model_name}_{variant}_{str(i).zfill(5)}"
            elif scenario.startswith('irr'):
                custom_id = f"{category}_{scenario}_{tsname}_ratio_{int(drop_ratio*100)}_{data_name}_{model_name}_{variant}_{str(i).zfill(5)}"

        # Skip already processed files
        if custom_id in results:
            continue
        
        # print(custom_id)
        # Perform anomaly detection with exponential backoff
        for attempt in range(num_retries):
            try:
                start_time = time.time()
                request = request_func(
                    # eval_dataset.series[i - 1],
                    train_dataset,
                    (category, scenario, tsname, dim, drop_ratio, data_name, i)
                )
                response = call_mllm(model_name, model, request, device)
                end_time = time.time()
                elasped_time = f'{end_time - start_time}s'
                # Write the result to jsonl
                with open(jsonl_fn, 'a') as f:
                    json.dump({'custom_id': custom_id, 'request': process_request(request), 'response': response, 'time': elasped_time}, f)
                    f.write('\n')
                # If successful, break the retry loop
                break
            except Exception as e:
                if "503" in str(e):  # Server not up yet, sleep until the server is up again
                    while True:
                        logger.debug("503 error, sleep 30 seconds")
                        time.sleep(30)
                        try:
                            response = call_mllm(model_name, model, request, device)
                            break
                        except Exception as e:
                            if "503" not in str(e):
                                break
                else:
                    logger.error(e)
                    # If an exception occurs, wait and then retry
                    wait_time = 2 ** (attempt + 3)
                    logger.debug(f"Attempt {attempt + 1} failed. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
        else:
            logger.error(f"Failed to process {custom_id} after {num_retries} attempts")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process online API anomaly detection.')
    parser.add_argument('--variant', type=str, default='0shot-vision', help='Variant type')
    parser.add_argument('--model_name', type=str, default='llama3-llava-next-8b', choices=['gpt-4o', 'gpt-4o-mini', 'gemini-1.5-pro', 'gemini-1.5-flash',
                                                                                      'llama3-llava-next-8b', 'llava-next-72b',
                                                                                      'Qwen2-VL-7B-Instruct', 'Qwen2-VL-72B-Instruct'], help='Model name')
    parser.add_argument("--category", type=str, default='synthetic', choices=['synthetic', 'semi'])
    parser.add_argument("--scenario", type=str, default='univariate', choices=['univariate', 'multivariate', 'irr_univariate', 'irr_multivariate'])
    parser.add_argument("--tsname", type=str, default=None, choices=['Symbols', 'ArticularyWordRecognition'])
    parser.add_argument("--data", type=str, default='global', choices=['global', 'contextual', 'seasonal', 'trend', 'shapelet', 
                                                                               'triangle', 'square', 'sawtooth',  'random_walk'], help="Synthesized anomaly type2")
    parser.add_argument("--drop_ratio", type=float, default=0.00)
    parser.add_argument("--dim", type=int, default=9)
    parser.add_argument("--device", type=str, default='auto')

    return parser.parse_args()

def main():
    args = parse_arguments()
    api_configs = create_api_configs()
    AD_with_retries(
        model_name=args.model_name,
        category=args.category,
        scenario=args.scenario,
        tsname=args.tsname,
        data_name=args.data,
        request_func=api_configs[args.variant],
        variant=args.variant,
        dim=args.dim,
        drop_ratio=args.drop_ratio,
        device=args.device
    )

if __name__ == '__main__':
    main()
