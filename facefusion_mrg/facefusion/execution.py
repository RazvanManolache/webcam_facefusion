import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
from functools import lru_cache
from typing import List, Optional

from onnxruntime import get_available_providers, set_default_logger_severity

import facefusion.choices
from facefusion.types import ExecutionDevice, ExecutionProvider, InferenceSessionProvider, ValueAndUnit

set_default_logger_severity(3)


def has_execution_provider(execution_provider : ExecutionProvider) -> bool:
	return execution_provider in get_available_execution_providers()


def get_available_execution_providers() -> List[ExecutionProvider]:
	inference_session_providers = get_available_providers()
	available_execution_providers : List[ExecutionProvider] = []

	for execution_provider, execution_provider_value in facefusion.choices.execution_provider_set.items():
		if execution_provider_value in inference_session_providers:
			index = facefusion.choices.execution_providers.index(execution_provider)
			available_execution_providers.insert(index, execution_provider)

	return available_execution_providers


def create_inference_session_providers(execution_device_id : str, execution_providers : List[ExecutionProvider], module_name : str = '', model_path : str = '') -> List[InferenceSessionProvider]:
	inference_session_providers : List[InferenceSessionProvider] = []

	for execution_provider in execution_providers:
		if execution_provider == 'cuda':
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
			{
				'device_id': execution_device_id,
				'cudnn_conv_algo_search': resolve_cudnn_conv_algo_search()
			}))
		if execution_provider == 'tensorrt':
			tensorrt_options = create_tensorrt_options(execution_device_id, module_name, model_path)
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
				tensorrt_options))
		if execution_provider in [ 'directml', 'rocm' ]:
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
			{
				'device_id': execution_device_id
			}))
		if execution_provider == 'migraphx':
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
			{
				'device_id': execution_device_id,
				'migraphx_model_cache_dir': '.caches'
			}))
		if execution_provider == 'openvino':
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
			{
				'device_type': resolve_openvino_device_type(execution_device_id),
				'precision': 'FP32'
			}))
		if execution_provider == 'coreml':
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
			{
				'SpecializationStrategy': 'FastPrediction',
				'ModelCacheDirectory': '.caches'
			}))

	# TensorRT does not cover every ONNX operator. Keep unsupported work on the
	# GPU where possible instead of letting ONNX Runtime jump straight to CPU.
	if 'tensorrt' in execution_providers and 'cuda' not in execution_providers:
		inference_session_providers.append((facefusion.choices.execution_provider_set.get('cuda'),
		{
			'device_id': execution_device_id,
			'cudnn_conv_algo_search': resolve_cudnn_conv_algo_search()
		}))

	if 'cpu' in execution_providers or 'tensorrt' in execution_providers:
		inference_session_providers.append(facefusion.choices.execution_provider_set.get('cpu'))

	return inference_session_providers


def create_tensorrt_options(execution_device_id : str, module_name : str, model_path : str) -> dict:
	# DFM faces are not numerically stable in FP16 (large image and mask drift),
	# while XSeg was measured at ~5x faster with negligible boundary changes.
	# Keep TensorRT FP16 deliberately limited to the occlusion-mask network.
	model_name = os.path.splitext(os.path.basename(model_path))[0] or 'model'
	module_parts = module_name.split('.')
	module_key = module_parts[-2] if len(module_parts) > 1 and module_parts[-1] == 'core' else (module_parts[-1] if module_parts else 'module')
	use_fp16 = module_key == 'face_masker' and model_name.startswith('xseg_')
	precision = 'fp16' if use_fp16 else 'fp32'
	cache_key = re.sub(r'[^a-zA-Z0-9_.-]+', '_', module_key + '_' + model_name + '_' + precision)
	cache_path = os.path.abspath(os.path.join('.caches', 'tensorrt', cache_key))
	os.makedirs(cache_path, exist_ok = True)
	tensorrt_options =\
	{
		'device_id': execution_device_id,
		'trt_fp16_enable': use_fp16,
		'trt_engine_cache_enable': True,
		'trt_engine_cache_path': cache_path,
		'trt_timing_cache_enable': True,
		'trt_timing_cache_path': cache_path,
		# Level 0 built these realtime models in seconds. Level 5 can spend
		# several minutes exploring tactics without a useful live-FPS gain.
		'trt_builder_optimization_level': 0
	}

	# XSeg has a dynamic batch dimension. A fixed realtime profile avoids a
	# generic engine and allows TensorRT to optimize exactly for one camera face.
	if use_fp16:
		profile_shape = 'input:1x256x256x3'
		tensorrt_options.update(
		{
			'trt_profile_min_shapes': profile_shape,
			'trt_profile_opt_shapes': profile_shape,
			'trt_profile_max_shapes': profile_shape
		})
	return tensorrt_options


def resolve_cudnn_conv_algo_search() -> str:
	execution_devices = detect_static_execution_devices()
	product_names = ('GeForce GTX 1630', 'GeForce GTX 1650', 'GeForce GTX 1660')

	for execution_device in execution_devices:
		if execution_device.get('product').get('name').startswith(product_names):
			return 'DEFAULT'

	return 'EXHAUSTIVE'


def resolve_openvino_device_type(execution_device_id : str) -> str:
	if execution_device_id == '0':
		return 'GPU'
	return 'GPU.' + execution_device_id


def run_nvidia_smi() -> subprocess.Popen[bytes]:
	commands = [ shutil.which('nvidia-smi'), '--query', '--xml-format' ]
	return subprocess.Popen(commands, stdout = subprocess.PIPE)


@lru_cache()
def detect_static_execution_devices() -> List[ExecutionDevice]:
	return detect_execution_devices()


def detect_execution_devices() -> List[ExecutionDevice]:
	execution_devices : List[ExecutionDevice] = []

	try:
		output, _ = run_nvidia_smi().communicate()
		root_element = ElementTree.fromstring(output)
	except Exception:
		root_element = ElementTree.Element('xml')

	for gpu_element in root_element.findall('gpu'):
		execution_devices.append(
		{
			'driver_version': root_element.findtext('driver_version'),
			'framework':
			{
				'name': 'CUDA',
				'version': root_element.findtext('cuda_version')
			},
			'product':
			{
				'vendor': 'NVIDIA',
				'name': gpu_element.findtext('product_name').replace('NVIDIA', '').strip()
			},
			'video_memory':
			{
				'total': create_value_and_unit(gpu_element.findtext('fb_memory_usage/total')),
				'free': create_value_and_unit(gpu_element.findtext('fb_memory_usage/free'))
			},
			'temperature':
			{
				'gpu': create_value_and_unit(gpu_element.findtext('temperature/gpu_temp')),
				'memory': create_value_and_unit(gpu_element.findtext('temperature/memory_temp'))
			},
			'utilization':
			{
				'gpu': create_value_and_unit(gpu_element.findtext('utilization/gpu_util')),
				'memory': create_value_and_unit(gpu_element.findtext('utilization/memory_util'))
			}
		})

	return execution_devices


def create_value_and_unit(text : str) -> Optional[ValueAndUnit]:
	if ' ' in text:
		value, unit = text.split()

		return\
		{
			'value': int(value),
			'unit': str(unit)
		}
	return None
