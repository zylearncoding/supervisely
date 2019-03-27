# coding: utf-8
from copy import deepcopy

from legacy_supervisely_lib import logger
from legacy_supervisely_lib.tasks import task_paths
from legacy_supervisely_lib.utils.general_utils import function_wrapper
from legacy_supervisely_lib.utils import config_readers
from legacy_supervisely_lib.utils import json_utils

from legacy_supervisely_lib.worker_api.rpc_servicer import AgentRPCServicer
from legacy_supervisely_lib.worker_api.agent_rpc import SimpleCache


class ModelDeploy:

    settings = {
        'cache_limit': 500,
        'connection': {
            'server_address': None,
            'token': None,
            'task_id': None,
        },
        'model_settings': {
        }
    }

    def __init__(self, model_applier_cls):
        self.model_applier_cls = model_applier_cls
        self.load_settings()

    def load_settings(self):
        self.settings = deepcopy(ModelDeploy.settings)
        new_settings = json_utils.json_load(task_paths.TaskPaths(determine_in_project=False).settings_path)
        logger.info('Input settings', extra={'settings': new_settings})
        config_readers.update_recursively(self.settings, new_settings)
        logger.info('Full settings', extra={'settings': self.settings})

    def run(self):
        model_settings = self.settings.get('model_settings', {})
        model_applier = function_wrapper(self.model_applier_cls, settings=model_settings)

        image_cache = SimpleCache(self.settings['cache_limit'])
        serv_instance = AgentRPCServicer(logger=logger,
                                         model_applier=model_applier,
                                         conn_settings=self.settings['connection'],
                                         cache=image_cache)
        serv_instance.run_inf_loop()
