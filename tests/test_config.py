import os
import importlib

def test_import_config_module():
    mod = importlib.import_module("src.config")
    assert hasattr(mod, "__file__")

def test_expected_paths_exist_or_defined():
    mod = importlib.import_module("src.config")
    # No exigimos que existan físicamente (DVC los trae), sólo que estén definidos
    assert hasattr(mod, "RAW_FILEPATH") or hasattr(mod, "RAW_DATA_PATH")
    assert hasattr(mod, "PROCESSED_FILEPATH") or hasattr(mod, "PROCESSED_DATA_PATH")
