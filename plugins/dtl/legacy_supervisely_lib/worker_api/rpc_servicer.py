# coding: utf-8

import concurrent.futures
import traceback
from queue import Queue
import threading

from .agent_api import AgentAPI
from .agent_rpc import decode_image, download_image_from_remote, download_data_from_remote, \
    send_from_memory_generator
from ..worker_proto import worker_api_pb2 as api_proto
from ..utils.imaging import drop_image_alpha_channel
from ..utils.json_utils import json_dumps, json_loads
from ..utils.general_utils import function_wrapper, function_wrapper_nofail
from ..tasks.progress_counter import report_agent_rpc_ready


class SingleImageApplier:
    def apply_single_image(self, image, message):
        raise NotImplementedError()


class AgentRPCServicer:
    NETW_CHUNK_SIZE = 1048576
    QUEUE_MAX_SIZE = 2000  # Maximum number of in-flight requests to avoid exhausting server memory.

    def __init__(self, logger, model_applier: SingleImageApplier, conn_settings, cache):
        self.logger = logger
        self.api = AgentAPI(token=conn_settings['token'],
                            server_address=conn_settings['server_address'],
                            ext_logger=self.logger)
        self.api.add_to_metadata('x-task-id', conn_settings['task_id'])

        self.model_applier = model_applier
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.download_queue = Queue(maxsize=self.QUEUE_MAX_SIZE)
        self.inference_queue = Queue(maxsize=self.QUEUE_MAX_SIZE)
        self.image_cache = cache
        self.logger.info('Created AgentRPCServicer', extra=conn_settings)

    def _load_image_from_sly(self, req_id, image_hash, src_node_token):
        self.logger.trace('Will look for image.', extra={
            'request_id': req_id, 'image_hash': image_hash, 'src_node_token': src_node_token
        })
        img_data = self.image_cache.get(image_hash)
        if img_data is None:
            img_data_packed = download_image_from_remote(self.api, image_hash, src_node_token, self.logger)
            img_data = decode_image(img_data_packed)
            self.image_cache.add(image_hash, img_data)

        return img_data

    def _load_arbitrary_image(self, req_id):
        self.logger.trace('Will load arbitrary image.', extra={'request_id': req_id})
        img_data_packed = download_data_from_remote(self.api, req_id, self.logger)
        img_data = decode_image(img_data_packed)
        return img_data

    def _load_data(self, event_obj):
        req_id = event_obj['request_id']
        image_hash = event_obj['data'].get('image_hash')
        if image_hash is None:
            img_data = self._load_arbitrary_image(req_id)
        else:
            src_node_token = event_obj['data'].get('src_node_token', '')
            img_data = self._load_image_from_sly(req_id, image_hash, src_node_token)

        # cv2.imwrite('/sly_task_data/last_loaded.png', img_data[:, :, ::-1])  # @TODO: rm debug
        event_obj['data']['image_arr'] = img_data
        self.inference_queue.put(item=(event_obj['data'], req_id))
        self.logger.trace('Input image obtained.', extra={'request_id': req_id})

    def _send_data(self, out_msg, req_id):
        self.logger.trace('Will send output data.', extra={'request_id': req_id})
        out_bytes = json_dumps(out_msg).encode('utf-8')

        self.api.put_stream_with_data('SendGeneralEventData',
                                      api_proto.Empty,
                                      send_from_memory_generator(out_bytes, self.NETW_CHUNK_SIZE),
                                      addit_headers={'x-request-id': req_id})
        self.logger.trace('Output data is sent.', extra={'request_id': req_id})

    def _single_img_inference(self, in_msg):
        img = in_msg['image_arr']
        if len(img.shape) != 3 or img.shape[2] not in [3, 4]:
            raise RuntimeError('Expect 3- or 4-channel image RGB(RGBA) [0..255].')
        elif img.shape[2] == 4:
            img = drop_image_alpha_channel(img)

        # may fail, it will be swallowed
        res = self.model_applier.apply_single_image(image=img, message=in_msg)
        return res

    def _sequential_inference(self):
        while True:
            in_msg, req_id = self.inference_queue.get(block=True, timeout=None)
            res_msg = {}
            try:
                res_msg.update(self._single_img_inference(in_msg))
                res_msg.update({'success': True})
            except Exception as e:
                self.logger.error(traceback.format_exc(), exc_info=True, extra={'exc_str': str(e)})
                res_msg.update({'success': False, 'error': str(e)})

            self.thread_pool.submit(function_wrapper_nofail, self._send_data, res_msg, req_id)  # skip errors

    def _load_data_loop(self):
        while True:
            event_obj = self.download_queue.get(block=True, timeout=None)
            function_wrapper_nofail(self._load_data, event_obj)

    def run_inf_loop(self):
        def seq_inf_wrapped():
            function_wrapper(self._sequential_inference)  # exit if raised

        load_data_thread = threading.Thread(target=self._load_data_loop, daemon=True)
        load_data_thread.start()
        inference_thread = threading.Thread(target=seq_inf_wrapped, daemon=True)
        inference_thread.start()
        report_agent_rpc_ready()

        for gen_event in self.api.get_endless_stream('GetGeneralEventsStream',
                                                     api_proto.GeneralEvent, api_proto.Empty()):
            data_str = gen_event.data.decode('utf-8')
            if len(data_str) == 0:
                data_str = '{}'

            event_obj = {
                'request_id': gen_event.request_id,
                'data': json_loads(data_str),
            }
            self.logger.debug('GET_INFERENCE_CALL', extra=event_obj)
            self.download_queue.put(event_obj, block=True)
