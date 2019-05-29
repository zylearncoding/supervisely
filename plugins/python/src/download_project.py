project_name = 'CHANGE_TO_YOUR_INPUT_PROJECT_NAME'

import os.path

sly.logger.info('DOWNLOAD_PROJECT', extra={'title': project_name})
project = api.project.get_info_by_name(WORKSPACE_ID, project_name)
sly.download_project(api, project.id, os.path.join(RESULT_ARTIFACTS_DIR, project_name), log_progress=True)
sly.logger.info('Project {!r} has been successfully downloaded.'.format(project_name))
