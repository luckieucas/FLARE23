import warnings
from functools import lru_cache

import numpy as np
import torch
from typing import Union, Tuple, List
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from scipy.ndimage import gaussian_filter
from torch import nn

from nnunetv2.utilities.helpers import empty_cache, dummy_context

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
 
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
 
def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    profile = builder.create_optimization_profile()
    network = builder.create_network()
    builder_config = builder.create_builder_config()
    builder_config.add_optimization_profile(profile)
    builder.max_batch_size = 1
    # use FP16 mode if possible
    builder_config.set_flag(trt.BuilderFlag.FP16)
    parser = trt.OnnxParser(network, TRT_LOGGER)
     
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')
        # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_engine(network,builder_config)
    #context = engine.create_execution_context()
    print("Completed creating Engine")
 
    return engine



@lru_cache(maxsize=1)
def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float16, device=torch.device('cuda', 0)) \
        -> torch.Tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map).type(dtype).to(device)

    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.type(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than tile_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is tile_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    num_steps[1] = 3
    num_steps[2] = 3
    max_step_value = image_size[0] - tile_size[0]
   # print(image_size, tile_size)
    if num_steps[0] > 1:
        actual_step_size = max_step_value / (num_steps[0] - 1)
    else:
        actual_step_size = 99999999999  # does not matter because there is only one step at 0
    steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[0])]
    steps.append(steps_here)

    #  决定 y 轴的steps,step个数固定为3
    if image_size[1] > 2.0 * tile_size[1]:
        steps_here = [int(np.round(image_size[1]/2)-1.0*tile_size[1]), int(np.round(image_size[1]/2)-0.5*tile_size[1]), int(np.round(image_size[1]/2)+0.*tile_size[1])]
    elif image_size[1] > tile_size[1]:
        actual_step_size = (image_size[1] - tile_size[1]) / 2
        steps_here = [int(np.round(actual_step_size * i)) for i in range(3)]
    else:
        steps_here = [0]
    steps.append(steps_here)

    # 准备 x 轴的 steps,step个数固定为3
    if image_size[2] > 2.0 * tile_size[2]:
        steps_here = [int(np.round(image_size[2]/2)-1.0*tile_size[2]), int(np.round(image_size[2]/2)-0.5*tile_size[2]), int(np.round(image_size[2]/2)+0.*tile_size[2])]
    elif image_size[2] > tile_size[2]:
        actual_step_size = (image_size[2] - tile_size[2]) / 2
        steps_here = [int(np.round(actual_step_size * i)) for i in range(3)]
    else:
        steps_here = [0]
    steps.append(steps_here)

  ##  for dim in range(len(tile_size)):
  ##      # the highest step value for this dimension is
  ##      max_step_value = image_size[dim] - tile_size[dim]
  ##      if num_steps[dim] > 1:
  ##          actual_step_size = max_step_value / (num_steps[dim] - 1)
  ##      else:
  ##          actual_step_size = 99999999999  # does not matter because there is only one step at 0

  ##      steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]


  ##      steps.append(steps_here)

    return steps


def get_sliding_window_generator(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float,
                                 verbose: bool = False):
    if len(tile_size) < len(image_size):
        assert len(tile_size) == len(image_size) - 1, 'if tile_size has less entries than image_size, len(tile_size) ' \
                                                      'must be one shorter than len(image_size) (only dimension ' \
                                                      'discrepancy of 1 allowed).'
        steps = compute_steps_for_sliding_window(image_size[1:], tile_size, tile_step_size)
        if verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicer = tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), tile_size)]])
                    yield slicer
    else:
        steps = compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)
     #   print('---------steps:{}--------'.format(steps))
        if verbose: print(f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), tile_size)]])
                    if len(steps[1])==3 and len(steps[2]) ==3 and sy ==steps[1][1] and sz == steps[2][1]:
                        yield [slicer,1]
                    else:
                        yield [slicer,0]



def maybe_mirror_and_predict(network: nn.Module, x: torch.Tensor, mirror_axes: Tuple[int, ...] = None) \
        -> torch.Tensor:
    prediction = network(x)
    
    # try use tensorRT
    # initialize TensorRT engine and parse ONNX model
    ONNX_FILE_PATH = "/data1/liupeng/Flare2023/nnunet.onnx"
    engine, context = build_engine(ONNX_FILE_PATH)
     # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize     # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
        # preprocess input data
    host_input = np.array(x.cpu().numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)
    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()
    # postprocess results
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    print(f"output shape: {output_data.shape}")
    

    if mirror_axes is not None:
        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2,))), (2,))
        if 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3,))), (3,))
        if 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (4,))), (4,))
        if 0 in mirror_axes and 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3))), (2, 3))
        if 0 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 4))), (2, 4))
        if 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3, 4))), (3, 4))
        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
        prediction /= num_predictons
    return prediction


def predict_sliding_window_return_logits(network: nn.Module,
                                         input_image: Union[np.ndarray, torch.Tensor],
                                         num_segmentation_heads: int,
                                         tile_size: Union[Tuple[int, ...], List[int]],
                                         mirror_axes: Tuple[int, ...] = None,
                                         tile_step_size: float = 0.5,
                                         use_gaussian: bool = True,
                                         precomputed_gaussian: torch.Tensor = None,
                                         perform_everything_on_gpu: bool = True,
                                         verbose: bool = True,
                                         device: torch.device = torch.device('cuda')) -> Union[np.ndarray, torch.Tensor]:
    if perform_everything_on_gpu:
        assert device.type == 'cuda', 'Can use perform_everything_on_gpu=True only when device="cuda"'

    network = network.to(device)
    network.eval()

    empty_cache(device)

    with torch.no_grad():
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
            assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if not torch.cuda.is_available():
                if perform_everything_on_gpu:
                    print('WARNING! "perform_everything_on_gpu" was True but cuda is not available! Set it to False...')
                perform_everything_on_gpu = False

            results_device = device if perform_everything_on_gpu else torch.device('cpu')

            if verbose: print("step_size:", tile_step_size)
            if verbose: print("mirror_axes:", mirror_axes)

            if not isinstance(input_image, torch.Tensor):
                # pytorch will warn about the numpy array not being writable. This doesnt matter though because we
                # just want to read it. Suppress the warning in order to not confuse users...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    input_image = torch.from_numpy(input_image)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, tile_size, 'constant', {'value': 0}, True, None)

            if use_gaussian:
                gaussian = compute_gaussian(tuple(tile_size), sigma_scale=1. / 8, value_scaling_factor=1000,
                                            device=device) if precomputed_gaussian is None else precomputed_gaussian

            slicers = get_sliding_window_generator(data.shape[1:], tile_size, tile_step_size, verbose=verbose)

            # preallocate results and num_predictions. Move everything to the correct device
            try:
                predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                               device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=results_device)
                gaussian = gaussian.to(results_device)
            except RuntimeError:
                # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                results_device = torch.device('cpu')
                predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                               device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=results_device)
                gaussian = gaussian.to(results_device)
            finally:
                empty_cache(device)
            ### save slicers
            slicers_list = []
            flag_list = []
            for sl in slicers:
                slicers_list.append(sl[0])
                flag_list.append(sl[1])
          #  print('---------------num patches:{}/{}--------------'.format(len(flag_list),np.sum(flag_list)))

            if np.sum(flag_list) > 0: ### (3,3) exist in the img
                ## check the z num
                z_num = len(flag_list) // 9
                for z_idx in range(z_num):
                    tmp_slicers = slicers_list[z_idx*9:(z_idx+1)*9]
                    workon = data[tmp_slicers[4]][None]
                    workon = workon.to(device, non_blocking=False)
                    prediction = maybe_mirror_and_predict(network, workon, mirror_axes)[0].to(results_device)
                    predicted_logits[tmp_slicers[4]] += (prediction * gaussian if use_gaussian else prediction)
                    n_predictions[tmp_slicers[4][1:]] += (gaussian if use_gaussian else 1)
                    is_empty = torch.sum(torch.argmax(prediction, 0)>0)<1000
                #    print(tile_size,prediction.shape)
                    #is_half_empty = torch.sum(torch.argmax(prediction[:,:int(0.6*tile_size[0]),:,:], 0)>0)<500
                    #if is_empty or is_half_empty:
                    #    print('----------------skip--------------------------:{}'.format(z_idx))

                    #if not (is_empty or is_half_empty):
                    if not is_empty:
                        for idx, sl in enumerate(tmp_slicers):
                            if idx == 4:
                                continue
                            workon = data[sl][None]
                            workon = workon.to(device, non_blocking=False)
                            prediction = maybe_mirror_and_predict(network, workon, mirror_axes)[0].to(results_device)
                            predicted_logits[sl] += (prediction * gaussian if use_gaussian else prediction)
                            n_predictions[sl[1:]] += (gaussian if use_gaussian else 1)

            else:
                
                for sl in slicers_list:
                    workon = data[sl][None]
                    workon = workon.to(device, non_blocking=False)

                    prediction = maybe_mirror_and_predict(network, workon, mirror_axes)[0].to(results_device)

                    predicted_logits[sl] += (prediction * gaussian if use_gaussian else prediction)
                    n_predictions[sl[1:]] += (gaussian if use_gaussian else 1)

            predicted_logits /= n_predictions
    empty_cache(device)
    return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]


if __name__ == '__main__':
    a = torch.rand((4, 2, 32, 23))
    a_npy = a.numpy()

    a_padded = pad_nd_image(a, new_shape=(13, 27))
    a_npy_padded = pad_nd_image(a_npy, new_shape=(13, 27))
    assert all([i == j for i, j in zip(a_padded.shape, (4, 2, 48, 27))])
    assert all([i == j for i, j in zip(a_npy_padded.shape, (4, 2, 48, 27))])
    assert np.all(a_padded.numpy() == a_npy_padded)
