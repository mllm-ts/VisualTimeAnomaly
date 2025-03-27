from prompt import create_openai_request

def create_api_configs():
    return {
        '0shot-vision': lambda train_dataset, data_tuple: create_openai_request(
            vision=True,
            few_shots=train_dataset.few_shots(num_shots=0),
            data_tuple=data_tuple
        )
    }
