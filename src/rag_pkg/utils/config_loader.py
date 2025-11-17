from typing import Any
import yaml
from rag_pkg.utils.path import DATA_CONFIG_PATH


def load_config(path: str) -> dict[str, Any]:
    config = {}
    with open(path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def load_all_configs(data_type="HPMC"):

    configs = {
        "data": load_config(path=DATA_CONFIG_PATH),
    }

    return configs


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dump_yaml(file_path: str) -> None:
    return yaml.dump(file_path, sort_keys=False, allow_unicode=True)


