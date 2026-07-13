from pathlib import Path
import yaml
from dotenv import load_dotenv
import os


CONFIG_PATH = Path(__file__).parent.parent / 'config/config.yaml'

def load_config() -> dict:
    load_dotenv()
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    for model in data['models']:
        for k, v in model.items():
            if isinstance(v, str) and v.startswith("$"):
                model[k] = os.environ[v[1:]]
    return data


if __name__ == '__main__':
    print(load_config())
