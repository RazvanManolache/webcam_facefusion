import importlib
import random
from time import perf_counter, sleep, time
from typing import Any, Callable, List, Optional

from onnxruntime import InferenceSession

from facefusion import logger, process_manager, state_manager, translator
from facefusion.app_context import detect_app_context
from facefusion.common_helper import is_windows
from facefusion.execution import create_inference_session_providers, has_execution_provider
from facefusion.exit_helper import fatal_exit
from facefusion.filesystem import get_file_name, is_file
from facefusion.time_helper import calculate_end_time
from facefusion.types import DownloadSet, ExecutionProvider, InferencePool, InferencePoolSet

INFERENCE_POOL_SET : InferencePoolSet =\
{
	'cli': {},
	'ui': {}
}

INFERENCE_TIMING_CALLBACK : Optional[Callable[[str, str, str, float], None]] = None


class TimedInferenceSession(InferenceSession):
	"""ONNX session that preserves the public type while reporting run time."""

	def __init__(self, model_path : str, providers : list, module_name : str, model_name : str):
		super().__init__(model_path, providers = providers)
		self._module_name = module_name
		self._model_name = model_name

	def run(self, *args : Any, **kwargs : Any) -> Any:
		started_at = perf_counter()
		try:
			return super().run(*args, **kwargs)
		finally:
			self._report_timing(started_at)

	def run_with_iobinding(self, *args : Any, **kwargs : Any) -> Any:
		started_at = perf_counter()
		try:
			return super().run_with_iobinding(*args, **kwargs)
		finally:
			self._report_timing(started_at)

	def _report_timing(self, started_at : float) -> None:
		callback = INFERENCE_TIMING_CALLBACK
		if callback:
			try:
				providers = self.get_providers()
				provider = providers[0] if providers else 'UnknownExecutionProvider'
				callback(self._module_name, self._model_name, provider, (perf_counter() - started_at) * 1000.0)
			except Exception:
				pass


def set_inference_timing_callback(callback : Optional[Callable[[str, str, str, float], None]]) -> None:
	global INFERENCE_TIMING_CALLBACK
	INFERENCE_TIMING_CALLBACK = callback


def get_inference_pool(module_name : str, model_names : List[str], model_source_set : DownloadSet) -> InferencePool:
	while process_manager.is_checking():
		sleep(0.5)
	execution_device_ids = state_manager.get_item('execution_device_ids')
	execution_providers = resolve_execution_providers(module_name)
	app_context = detect_app_context()

	for execution_device_id in execution_device_ids:
		inference_context = get_inference_context(module_name, model_names, execution_device_id, execution_providers)

		if app_context == 'cli' and INFERENCE_POOL_SET.get('ui').get(inference_context):
			INFERENCE_POOL_SET['cli'][inference_context] = INFERENCE_POOL_SET.get('ui').get(inference_context)
		if app_context == 'ui' and INFERENCE_POOL_SET.get('cli').get(inference_context):
			INFERENCE_POOL_SET['ui'][inference_context] = INFERENCE_POOL_SET.get('cli').get(inference_context)
		if not INFERENCE_POOL_SET.get(app_context).get(inference_context):
			INFERENCE_POOL_SET[app_context][inference_context] = create_inference_pool(module_name, model_source_set, execution_device_id, execution_providers)

	current_inference_context = get_inference_context(module_name, model_names, random.choice(execution_device_ids), execution_providers)
	return INFERENCE_POOL_SET.get(app_context).get(current_inference_context)


def create_inference_pool(module_name : str, model_source_set : DownloadSet, execution_device_id : str, execution_providers : List[ExecutionProvider]) -> InferencePool:
	inference_pool : InferencePool = {}

	for model_name in model_source_set.keys():
		model_path = model_source_set.get(model_name).get('path')
		if is_file(model_path):
			inference_pool[model_name] = create_inference_session(model_path, execution_device_id, execution_providers, module_name, model_name)

	return inference_pool


def clear_inference_pool(module_name : str, model_names : List[str]) -> None:
	execution_device_ids = state_manager.get_item('execution_device_ids')
	execution_providers = resolve_execution_providers(module_name)
	app_context = detect_app_context()

	if is_windows() and has_execution_provider('directml'):
		INFERENCE_POOL_SET[app_context].clear()

	for execution_device_id in execution_device_ids:
		inference_context = get_inference_context(module_name, model_names, execution_device_id, execution_providers)
		if INFERENCE_POOL_SET.get(app_context).get(inference_context):
			del INFERENCE_POOL_SET[app_context][inference_context]


def create_inference_session(model_path : str, execution_device_id : str, execution_providers : List[ExecutionProvider], module_name : str = '', model_name : str = '') -> InferenceSession:
	model_file_name = get_file_name(model_path)
	start_time = time()

	try:
		inference_session_providers = create_inference_session_providers(execution_device_id, execution_providers, module_name, model_path)
		inference_session = TimedInferenceSession(model_path, inference_session_providers, module_name, model_name or model_file_name)
		logger.debug(translator.get('loading_model_succeeded').format(model_name = model_file_name, seconds = calculate_end_time(start_time)), __name__)
		return inference_session

	except Exception:
		logger.error(translator.get('loading_model_failed').format(model_name = model_file_name), __name__)
		fatal_exit(1)


def get_inference_context(module_name : str, model_names : List[str], execution_device_id : str, execution_providers : List[ExecutionProvider]) -> str:
	inference_context = '.'.join([ module_name ] + model_names + [ execution_device_id ] + list(execution_providers))
	return inference_context


def resolve_execution_providers(module_name : str) -> List[ExecutionProvider]:
	module = importlib.import_module(module_name)

	if hasattr(module, 'resolve_execution_providers'):
		return getattr(module, 'resolve_execution_providers')()
	return state_manager.get_item('execution_providers')
