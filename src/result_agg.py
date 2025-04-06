import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from utils import (
    interval_to_vector,
    point_to_vector,
    id_to_vector
)
import pickle
import os
from dataloader import TSIDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events


def df_to_latex(df):
    # Step 1: Process the index to extract only the model name part
    df = df.reset_index()  # Reset index to bring it as a column
    df['index'] = df['index'].str.split(' ').str[0]  # Keep only the model name
    df.rename(columns={'index': 'model'}, inplace=True)  # Rename the index column
    
    # Step 2: Sort the DataFrame by a custom order for models
    order = {"gpt-4o": 0, "gpt-4o-mini": 1, "gemini-1.5-pro": 2, "gemini-1.5-flash": 3, 
             "llava-next-72b": 4, "llama3-llava-next-8b": 5,
             'Qwen2-VL-72B-Instruct': 6, 'Qwen2-VL-7B-Instruct': 7}
    df['priority'] = df['model'].apply(lambda x: order.get(x.lower(), 20))  # Default priority for others is 4
    df = df.sort_values(by=['priority', 'model']).drop(columns=['priority'])  # Sort and drop priority column
    
    # Step 3: Truncate numerical values to the first 3 decimal digits
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col] = df[col].apply(lambda x: f"{x*100:.2f}")  # Truncate to 4 decimals

    # Step 4: Convert to plain LaTeX table
    latex_table = df.to_latex(index=False, column_format="|l" + "r" * (len(df.columns) - 1) + "|")

    return df, latex_table

def compute_metrics(gt, prediction):
    if np.count_nonzero(gt) == 0:
        print('ground truth is all zero!!!')
        exit()
    elif np.count_nonzero(prediction) == 0:
        metrics = {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'affi precision': 0,
            'affi recall': 0,
            'affi f1': 0
        }
    else:
        precision = precision_score(gt, prediction)
        recall = recall_score(gt, prediction)
        f1 = f1_score(gt, prediction)
        
        events_pred = convert_vector_to_events(prediction)
        events_gt = convert_vector_to_events(gt)
        Trange = (0, len(prediction))
        aff = pr_from_events(events_pred, events_gt, Trange)
        
        # Calculate affiliation F1
        if aff['precision'] + aff['recall'] == 0:
            affi_f1 = 0
        else:
            affi_f1 = 2 * (aff['precision'] * aff['recall']) / (aff['precision'] + aff['recall'])
        
        metrics = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'affi precision': round(aff['precision'], 3),
            'affi recall': round(aff['recall'], 3),
            'affi f1': round(affi_f1, 3)
        }
    return metrics

def compute_metrics_for_results(eval_dataset, results, scenario, data_name, num_samples=100):
    metric_names = [
        "precision",
        "recall",
        "f1",
        "affi precision",
        "affi recall",
        "affi f1",
    ]
    results_dict = {key: [[] for _ in metric_names] for key in results.keys()}

    for i in trange(0, num_samples):
        anomaly_locations, series = eval_dataset[i][0].numpy(), eval_dataset[i][1].numpy()
        if scenario.endswith('univariate'):
            len_series = series.shape[0]
        elif scenario.endswith('multivariate'):
            dim = series.shape[1]
        
        if data_name in ['global', 'contextual']:
            gt = point_to_vector(anomaly_locations, len_vector=len_series)
        elif data_name in ['seasonal', 'trend', 'shapelet']:
            gt = interval_to_vector(anomaly_locations, start=0, end=len_series)
        else:
            gt = id_to_vector(anomaly_locations, dim)

        for name, prediction in results.items():
            if prediction[i] == None:
                continue
            
            if data_name in ['global', 'contextual']:
                pred = point_to_vector(prediction[i], len_vector=len_series)
            elif data_name in ['seasonal', 'trend', 'shapelet']:
                pred = interval_to_vector(prediction[i], start=0, end=len_series, pred=True)
            else:
                pred = id_to_vector(prediction[i], dim)

            if scenario == 'irr_univariate':
                drop_index = eval_dataset[i][2].numpy().astype(int)
                gt_irr = np.delete(gt, drop_index)
                pred_irr = np.delete(pred, drop_index)

            metrics = compute_metrics(gt, pred) if scenario != 'irr_univariate' else compute_metrics(gt_irr, pred_irr)

            for idx, metric_name in enumerate(metric_names):
                results_dict[name][idx].append(metrics[metric_name])

    df = pd.DataFrame(
        {k: np.mean(v, axis=1) for k, v in results_dict.items()},
        index=["precision", "recall", "f1", "affi precision", "affi recall", "affi f1"],
    )

    return df


def load_time_results(result_fn):
    import json

    with open(result_fn, 'r') as f:
        time_results = []
        for line in f:
            info = json.loads(line)
            try:
                time = float(info['time'][:-1])
                time_results.append(time)
            except Exception:
                time_results.append(None)
                continue
    
    return time_results

def parse_output(output: str, data_name: str) -> dict:
    """Parse the output of the AD model.

    Args:
        output: The output of the AD model.

    Returns:
        A dictionary containing the parsed output.
    """
    import json
    import re
    
    # handle cases where the max_tokens are reached  
    if output.count('[') == output.count(']') + 1:
        # remove invalid tokens
        if output.endswith(',') or output.endswith(' '):
            output = output.rstrip(', ')
        else:
            output = output.rstrip('0123456789').rstrip(', ')
        # Add the missing right bracket
        output += ']'

    # Trim the output string
    trimmed_output = output[output.index('['):output.rindex(']') + 1]

    # check if containing digits
    trimmed_output = '[]' if not re.search(r'\d', trimmed_output) else trimmed_output

    # Try to parse the output as JSON
    parsed_output = json.loads(trimmed_output)

    # Validate the output: list of dict with keys start and end
    if data_name in ['global', 'contextual', 'triangle', 'square', 'sawtooth',  'random_walk']:
        for item in parsed_output:
            if not isinstance(item, int):
                raise ValueError("Parsed output contains non-int items")
    else:
        for item in parsed_output:
            # if not isinstance(item, dict):
            #     raise ValueError("Parsed output contains non-dict items")
            # if 'start' not in item or 'end' not in item:
            #     raise ValueError("Parsed output dictionaries must contain 'start' and 'end' keys")
            if not isinstance(item, list):
                raise ValueError("Parsed output contains non-dict items")
    
    return parsed_output

def load_results(result_fn, data_name, raw=False, postprocess_func: callable = None):
    """
    Load and process results from a result JSON lines file.

    Parameters
    ----------
    result_fn : str
        The filename of the JSON lines file containing the results.
    raw : bool, optional
        If True, return raw JSON objects. If False, parse the response
        and convert it to a vector. Default is False.
    postprocess_func : callable, optional
        A function to postprocess the results (e.g., scaling down). Default is None.

    Returns
    -------
    list
        A list of processed results. Each item is either a raw JSON object
        or a vector representation of anomalies, depending on the
        `raw` parameter.

    Notes
    -----
    The function attempts to parse each line in the file. If parsing fails,
    it appends an empty vector to the results.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    JSONDecodeError
        If a line in the file is not valid JSON.
    """
    import json
    import pandas as pd
    
    if postprocess_func is None:
        postprocess_func = lambda x: x
    
    with open(result_fn, 'r') as f:
        results = []
        for line in f:
            info = json.loads(line)
            if raw:
                results.append(info)
            else:
                try:
                    response_parsed = parse_output(postprocess_func(info['response']), data_name)
                    results.append(response_parsed)
                except Exception:
                    results.append(None)
                    continue
    
    return results

def collect_results(directory, raw=False, ignore=[]):
    """
    Collect and process results from JSON lines files in a directory.

    Parameters
    ----------
    directory : str
        The path to the directory containing the JSON lines files.
    raw : bool, optional
        If True, return raw JSON objects. If False, parse the responses.
        Default is False.
    ignore: list[str], optional
        Skip folders containing these names. Default is an empty list.

    Returns
    -------
    dict
        A dictionary where keys are model names with variants, and values
        are lists of processed results from each file.

    Notes
    -----
    This function walks through the given directory, processing each
    `.jsonl` file except those with 'requests' in the filename. It uses
    the directory name as the model name and the filename (sans extension)
    as the variant.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.
    """
    import os
    from config import postprocess_configs

    results = {}
    config = postprocess_configs()
    for root, _, files in os.walk(directory):
        for file in files:
            skip = False
            for ignore_folder in ignore:
                if ignore_folder in root:
                    skip = True
                    break
            if skip:
                continue
            if 'requests' not in file and file.endswith('.jsonl'):
                model_name = os.path.basename(root)
                data_name = os.path.basename(os.path.dirname(root))
                # scenario = os.path.basename(os.path.dirname(os.path.dirname(root)))
                variant = file.replace('.jsonl', '')
                if variant in config:
                    pf = config[variant]
                else:
                    pf = None
                result_fn = os.path.join(root, file)
                model_key = f'{model_name} ({variant})'
                results[model_key] = load_results(result_fn, data_name, raw=raw, postprocess_func=pf)

    return results

def load_datasets(category, scenario, tsname, dim, drop_ratio, data_name):
    if category == 'synthetic':
        if scenario == 'univariate':
            data_dir = f"data/{category}/{scenario}/{data_name}/eval"
            train_dir = f"data/{category}/{scenario}/{data_name}/train"
        elif scenario == 'multivariate':
            data_dir = f"data/{category}/{scenario}/dim_{dim}/{data_name}/eval"
            train_dir = f"data/{category}/{scenario}/dim_{dim}/{data_name}/train"
        elif scenario.startswith('irr'):
            data_dir = f"data/{category}/{scenario}/ratio_{int(drop_ratio*100)}/{data_name}/eval"
            train_dir = f"data/{category}/{scenario}/ratio_{int(drop_ratio*100)}/{data_name}/train"
    elif category == 'semi':
        if scenario == 'univariate':
            data_dir = f"data/{category}/{scenario}/{tsname}/{data_name}/eval"
            train_dir = f"data/{category}/{scenario}/{tsname}/{data_name}/train"
        elif scenario == 'multivariate':
            data_dir = f"data/{category}/{scenario}/{tsname}/dim_{dim}/{data_name}/eval"
            train_dir = f"data/{category}/{scenario}/{tsname}/dim_{dim}/{data_name}/train"
        elif scenario.startswith('irr'):
            data_dir = f"data/{category}/{scenario}/{tsname}/ratio_{int(drop_ratio*100)}/{data_name}/eval"
            train_dir = f"data/{category}/{scenario}/{tsname}/ratio_{int(drop_ratio*100)}/{data_name}/train"
    eval_dataset = TSIDataset(data_dir)
    train_dataset = TSIDataset(train_dir)
    return eval_dataset, train_dataset


def main(args):
    category = args.category
    scenario = args.scenario
    tsname = args.tsname
    data_name = args.data_name
    drop_ratio = args.drop_ratio
    dim = args.dim
    label_name = args.label_name
    table_caption = args.table_caption
    
    eval_dataset, train_dataset = load_datasets(category, scenario, tsname, dim, drop_ratio, data_name)
    if category == 'synthetic':
        if scenario == 'univariate':
            directory = f"results/{category}/{scenario}/{data_name}"
        elif scenario == 'multivariate':
            directory = f"results/{category}/{scenario}/dim_{dim}/{data_name}"
        elif scenario.startswith('irr'):
            directory = f"results/{category}/{scenario}/ratio_{int(drop_ratio*100)}/{data_name}"
    elif category == 'semi':
        if scenario == 'univariate':
            directory = f'results/{category}/{scenario}/{tsname}/{data_name}'
        elif scenario == 'multivariate':
            directory = f"results/{category}/{scenario}/{tsname}/dim_{dim}/{data_name}"
        elif scenario.startswith('irr'):
            directory = f'results/{category}/{scenario}/{tsname}/ratio_{int(drop_ratio*100)}/{data_name}'
    results = collect_results(directory, ignore=[])

    df = compute_metrics_for_results(eval_dataset, results, scenario, data_name, num_samples=len(eval_dataset))
    df = df.T
    # print(df)
    df, latex_table = df_to_latex(df.copy())
    print(df)
    print(latex_table)

    if scenario.endswith('univariate'):
        df_selected = df[['model', 'affi precision', 'affi recall', 'affi f1']].rename(columns={'affi precision': 'precision', 'affi recall': 'recall', 'affi f1': 'f1'})\
            .set_index('model')
    elif scenario.endswith('multivariate'):
        df_selected = df[['model', 'precision', 'recall', 'f1']].set_index('model')

    # Attempt to drop the index, catch exception if it doesn't exist
    try:
        df_selected = df_selected.drop(index='gemini-1.5-flash-8b')
    except KeyError:
        pass  # If index does not exist, do nothing and proceed

    with open(f"{directory}/df.pkl", "wb") as f:
        pickle.dump(df_selected, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process time series data and generate LaTeX table.")
    parser.add_argument("--category", type=str, default='synthetic', choices=['synthetic', 'semi', 'real'])
    parser.add_argument("--scenario", type=str, default='univariate', choices=['univariate', 'multivariate', 'irr_univariate', 'irr_multivariate', 'long'])
    parser.add_argument("--tsname", type=str, default=None, choices=['Symbols', 'ArticularyWordRecognition'])
    parser.add_argument("--data_name", type=str, default='global', choices=['global', 'contextual', 'seasonal', 'trend', 'shapelet', 
                                                                               'triangle', 'square', 'sawtooth',  'random_walk',
                                                                               'long'], help="Synthesized anomaly type2")
    parser.add_argument("--drop_ratio", type=float, default=0.00)
    parser.add_argument("--dim", type=int, default=9)
    parser.add_argument("--label_name", type=str, default='trend-exp', help="Name of the experiment")
    parser.add_argument("--table_caption", type=str, default='Trend anomalies in shifting sine wave', help="Caption for the LaTeX table")
    args = parser.parse_args()
    main(args)
