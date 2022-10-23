import os
from pathlib import Path

# Project and source files
PROJECT_PATH = Path(__file__).parent.parent.parent
CONFIGS_PATH = PROJECT_PATH / 'configs'
DATASETS_PATH = PROJECT_PATH / 'datasets'
MODELS_PATH = PROJECT_PATH / 'models'
RUNS_PATH = PROJECT_PATH / 'runs'
TMP_PATH = Path(os.environ['SCRATCH']) if 'SCRATCH' in os.environ else None
if TMP_PATH is not None:
    os.environ['TORCH_HOME'] = os.environ['WORK'] + '/.torch'
