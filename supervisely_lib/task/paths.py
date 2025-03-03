# coding: utf-8
import os.path

ARTIFACTS = 'artifacts'
CODE = 'code'
CONFIG_JSON = 'config.json'
DATA = 'data'
GRAPH_JSON = 'graph.json'
MODEL = 'model'
PROJECTS = 'projects'
TASK_CONFIG_JSON = 'task_config.json'
TASK_SETTINGS_JSON = 'task_settings.json'
RESULTS = 'results'
TMP = 'tmp'


class TaskPaths:
    TASK_DIR = '/sly_task_data'
    SETTINGS_PATH = os.path.join(TASK_DIR, TASK_SETTINGS_JSON)  # Deprecated - use TASK_CONFIG_PATH instead
    TASK_CONFIG_PATH = os.path.join(TASK_DIR, TASK_CONFIG_JSON)
    DATA_DIR = os.path.join(TASK_DIR, DATA)
    RESULTS_DIR = os.path.join(TASK_DIR, RESULTS)
    DEBUG_DIR = os.path.join(TASK_DIR, TMP)
    GRAPH_PATH = os.path.join(TASK_DIR, GRAPH_JSON)
    MODEL_DIR = os.path.join(TASK_DIR, MODEL)
    MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, CONFIG_JSON)
    MODEL_CONFIG_NAME = CONFIG_JSON

    # For Python tasks only.
    OUT_ARTIFACTS_DIR = os.path.join(RESULTS_DIR, ARTIFACTS)
    OUT_PROJECTS_DIR = os.path.join(RESULTS_DIR, PROJECTS)
