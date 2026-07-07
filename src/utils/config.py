"""
统一配置加载器
"""

import yaml
from pathlib import Path


class Config:

    def __init__(self, path="config/system.yaml"):

        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError("config/system.yaml not found")

        with open(self.path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def get(self, *keys):

        value = self.data

        for k in keys:
            value = value[k]

        return value