import torch
from collections import OrderedDict
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import numpy as np
import onnx
from onnx_tf.backend import prepare

import networks
import onnxkeras_patch


_MEAN = (0.406, 0.456, 0.485)
_STD = (0.225, 0.224, 0.229)
_MODEL_HEIGHT = 473
_MODEL_WIDTH = 473


def _load_std_model(checkpoint_path):
    model_std = networks.init_model(
        'resnet101', num_classes=20, pretrained=None)
    state_dict = torch.load(checkpoint_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model_std.load_state_dict(new_state_dict)

    # Change model to eval mode and only forward necessary paths
    model_std.eval()
    model_std._only_parse_output = True
    return model_std


def _torch_to_onnx(model_std, onnx_path):
    inp_std = torch.zeros(1, 3, _MODEL_HEIGHT,
                          _MODEL_WIDTH, dtype=torch.float32)
    return torch.onnx.export(
        model_std, inp_std, onnx_path, export_params=True,
        do_constant_folding=False, verbose=False, input_names=["input"], output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, opset_version=11)


def _onnx_to_keras(onnx_path):
    model_onnx = onnx.load(onnx_path)
    return onnxkeras_patch.onnx_to_keras(model_onnx, ['input'], verbose=False, change_ordering=True)


def _get_affine_transform(input_h, input_w, output_h, output_w):
    input_h, input_w, output_h, output_w = [
        tf.cast(x, tf.float32) for x in (input_h, input_w, output_h, output_w)]
    aspect_ratio = output_w / output_h
    if input_h * aspect_ratio > input_w:
        scale = output_h / input_h
    else:
        scale = output_w / input_w

    center = tf.convert_to_tensor([input_w / 2., input_h / 2.])
    new_center = tf.convert_to_tensor([output_w / 2., output_h / 2.])
    translation = new_center - center * scale
    tx, ty = translation[0], translation[1]
    transform = tf.convert_to_tensor([scale, 0., tx, 0., scale, ty, 0., 0.])
    iscale = 1. / scale
    itransform = tf.convert_to_tensor(
        [iscale, 0., -tx * iscale, 0., iscale, -ty * iscale, 0., 0.])
    return transform, itransform


class ModelWithPreprocessing(keras.Model):
    def __init__(self, model, model_width=_MODEL_WIDTH, model_height=_MODEL_HEIGHT):
        super().__init__()
        self._model = model
        self._input_width = model_width
        self._input_height = model_height
        self._mean = tf.convert_to_tensor(_MEAN)
        self._std = tf.convert_to_tensor(_STD)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32)])
    def call(self, inputs):
        # Pad input image and normalize
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        transform, itransform = _get_affine_transform(
            h, w, self._input_height, self._input_width)
        padded_inputs = tfa.image.transform(
            inputs, itransform, output_shape=(self._input_height, self._input_width))
        padded_inputs = (padded_inputs - self._mean) / self._std

        outputs = self._model(padded_inputs)
        # Upsample outputs
        outputs = tf.image.resize(outputs, size=(
            self._input_height, self._input_width))
        # Unpad output to original image
        outputs = tfa.image.transform(outputs, transform, output_shape=(h, w))
        return tf.argmax(outputs, axis=-1)[..., None]


def convert_torch_to_saved_model(checkpoint_path, onnx_path, saved_model_path):
    model_std = _load_std_model(checkpoint_path)
    _torch_to_onnx(model_std, onnx_path)
    model_keras = _onnx_to_keras(onnx_path)
    model_prepro = ModelWithPreprocessing(model_keras)
    model_prepro.build(input_shape=(1, _MODEL_HEIGHT, _MODEL_WIDTH, 3))
    model_prepro.save(saved_model_path)
    return model_prepro
