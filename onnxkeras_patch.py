from onnx2keras.pooling_layers import *
from onnx2keras.reshape_layers import *
from tensorflow import keras
import onnx2keras
from onnx2keras.converter import *


def onnx_to_keras(onnx_model, input_names,
                  input_shapes=None, name_policy=None, verbose=True, change_ordering=False):
    """
    Convert ONNX graph to Keras model format
    :param onnx_model: loaded ONNX model
    :param input_names: list with input names
    :param input_shapes: override input shapes (experimental)
    :param name_policy: override layer names. None, "short" or "renumerate" (experimental)
    :param verbose: verbose output
    :param change_ordering: change ordering to HWC (experimental)
    :return: Keras model
    """
    # Use channels first format by default.
    keras_fmt = keras.backend.image_data_format()
    keras.backend.set_image_data_format('channels_first')

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger('onnx2keras')

    logger.info('Converter is called.')

    onnx_weights = onnx_model.graph.initializer
    onnx_inputs = onnx_model.graph.input
    onnx_outputs = [i.name for i in onnx_model.graph.output]
    onnx_nodes = onnx_model.graph.node

    logger.debug('List input shapes:')
    logger.debug(input_shapes)

    logger.debug('List inputs:')
    for i, input in enumerate(onnx_inputs):
        logger.debug('Input {0} -> {1}.'.format(i, input.name))

    logger.debug('List outputs:')
    for i, output in enumerate(onnx_outputs):
        logger.debug('Output {0} -> {1}.'.format(i, output))

    logger.debug('Gathering weights to dictionary.')
    weights = {}
    for onnx_w in onnx_weights:
        try:
            if len(onnx_w.ListFields()) < 4:
                onnx_extracted_weights_name = onnx_w.ListFields()[1][1]
            else:
                onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
            weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)
        except:
            onnx_extracted_weights_name = onnx_w.ListFields()[3][1]
            weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)

        logger.debug('Found weight {0} with shape {1}.'.format(
                     onnx_extracted_weights_name,
                     weights[onnx_extracted_weights_name].shape))

    layers = dict()
    lambda_funcs = dict()
    keras_outputs = []
    keras_inputs = []

    for i, input_name in enumerate(input_names):
        for onnx_i in onnx_inputs:
            if onnx_i.name == input_name:
                if input_shapes:
                    input_shape = input_shapes[i]
                else:
                    input_shape = [i.dim_value for i in onnx_i.type.tensor_type.shape.dim][1:]

                layers[input_name] = keras.layers.InputLayer(
                    input_shape=input_shape, name=input_name
                ).output

                keras_inputs.append(layers[input_name])

                logger.debug('Found input {0} with shape {1}'.format(input_name, input_shape))

    # Convert every operation separable
    node_names = []
    for node_index, node in enumerate(onnx_nodes):
        node_type = node.op_type
        node_params = onnx_node_attributes_to_dict(node.attribute)

        # Add global converter info:
        node_params['change_ordering'] = change_ordering
        node_params['name_policy'] = name_policy

        node_name = str(node.output[0])
        keras_names = []
        for output_index, output in enumerate(node.output):
            if name_policy == 'short':
                keras_name = keras_name_i = str(output)[:8]
                suffix = 1
                while keras_name_i in node_names:
                    keras_name_i = keras_name + '_' + str(suffix)
                    suffix += 1
                keras_names.append(keras_name_i)
            elif name_policy == 'renumerate':
                postfix = node_index if len(node.output) == 1 else "%s_%s" % (node_index, output_index)
                keras_names.append('LAYER_%s' % postfix)
            else:
                keras_names.append(output)

        if len(node.output) != 1:
            logger.warning('Trying to convert multi-output node')
            node_params['_outputs'] = list(node.output)
            node_names.extend(keras_names)
        else:
            keras_names = keras_names[0]
            node_names.append(keras_names)

        logger.debug('######')
        logger.debug('...')
        logger.debug('Converting ONNX operation')
        logger.debug('type: %s', node_type)
        logger.debug('node_name: %s', node_name)
        logger.debug('node_params: %s', node_params)
        logger.debug('...')

        logger.debug('Check if all inputs are available:')
        if len(node.input) == 0 and node_type != 'Constant':
            raise AttributeError('Operation doesn\'t have an input. Aborting.')

        for i, node_input in enumerate(node.input):
            logger.debug('Check input %i (name %s).', i, node_input)
            if node_input not in layers:
                logger.debug('The input not found in layers / model inputs.')

                if node_input in weights:
                    logger.debug('Found in weights, add as a numpy constant.')
                    layers[node_input] = weights[node_input]
                else:
                    raise AttributeError('Current node is not in weights / model inputs / layers.')
        else:
            logger.debug('... found all, continue')

        keras.backend.set_image_data_format('channels_first')
        AVAILABLE_CONVERTERS[node_type](
            node,
            node_params,
            layers,
            lambda_funcs,
            node_name,
            keras_names
        )
        if isinstance(keras_names, list):
            keras_names = keras_names[0]

        try:
            logger.debug('Output TF Layer -> ' + str(layers[keras_names]))
        except KeyError:
            pass

    # Check for terminal nodes
    for layer in onnx_outputs:
        if layer in layers:
            keras_outputs.append(layers[layer])

    # Create model
    model = keras.models.Model(inputs=keras_inputs, outputs=keras_outputs)

    if change_ordering:
        import numpy as np
        conf = model.get_config()

        for layer in conf['layers']:
            if layer['config'] and 'shared_axes' in layer['config']:
                # TODO: check axes first (if it's not 4D tensor)
                layer['config']['shared_axes'] = [1, 2]

            if layer['config'] and 'batch_input_shape' in layer['config']:
                layer['config']['batch_input_shape'] = \
                    tuple(np.reshape(np.array(
                        [
                            [None] +
                            list(layer['config']['batch_input_shape'][2:][:]) +
                            [layer['config']['batch_input_shape'][1]]
                        ]), -1
                    ))
            if layer['config'] and 'target_shape' in layer['config']:
                if len(list(layer['config']['target_shape'][1:][:])) > 0:
                    layer['config']['target_shape'] = \
                        tuple(np.reshape(np.array(
                                list(layer['config']['target_shape'][1:]) +
                                [layer['config']['target_shape'][0]]
                            ), -1),)

            if layer['config'] and 'data_format' in layer['config']:
                layer['config']['data_format'] = 'channels_last'
            if layer['config'] and 'axis' in layer['config']:
                axis = layer['config']['axis']
                if layer['config']['axis'] == 3:
                    layer['config']['axis'] = 1
                if layer['config']['axis'] == 1:
                    layer['config']['axis'] = 3
                if hasattr(axis, "__getitem__"):
                    assert len(axis) == 1
                    if axis[0] == 3:
                        layer['config']['axis'] = 1
                    if axis[0] == 1:
                        layer['config']['axis'] = 3

        for layer in conf['layers']:
            if 'function' in layer['config'] and layer['config']['function'][1] is not None:
                kerasf = list(layer['config']['function'])
                dargs = list(kerasf[1])
                func = lambda_funcs.get(layer['name'])

                if func:
                    if len(dargs) > 1:
                        params = inspect.signature(func).parameters
                        i = list(params.keys()).index('axes') if ('axes' in params) else -1

                        if i > 0:
                            i -= 1
                            axes = list(range(len(dargs[i].shape)))
                            axes = axes[0:1] + axes[2:] + axes[1:2]
                            dargs[i] = np.transpose(dargs[i], axes)

                        i = list(params.keys()).index('axis') if ('axis' in params) else -1

                        if i > 0:
                            i -= 1
                            axis = np.array(dargs[i])
                            axes_map = np.array([0, 3, 1, 2])
                            dargs[i] = axes_map[axis]
                    else:
                        if dargs[0] == -1:
                            dargs = [1]
                        elif dargs[0] == 3:
                            dargs = [1]

                kerasf[1] = tuple(dargs)
                layer['config']['function'] = tuple(kerasf)

        keras.backend.set_image_data_format('channels_last')
        model_tf_ordering = keras.models.Model.from_config(conf)

        for dst_layer, src_layer, conf in zip(model_tf_ordering.layers, model.layers, conf['layers']):
            W = src_layer.get_weights()
            # TODO: check axes first (if it's not 4D tensor)
            if conf['config'] and 'shared_axes' in conf['config']:
                W[0] = W[0].transpose(1, 2, 0)
            dst_layer.set_weights(W)

        model = model_tf_ordering

    keras.backend.set_image_data_format(keras_fmt)

    return model
    

def convert_slice(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert slice.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:slice')
    if is_numpy(layers[node.input[0]]):
        if params['change_ordering']:
            pass
        logger.debug('Slice numpy constants')
        if 'axes' in params:
            if len(params["axes"]) != 1:
                raise NotImplementedError("Multiple axes in Slice is not implemented")
            axes = params["axes"][0]
            ends = params["ends"][0]
            starts = params["starts"][0]
            if axes == 0:
                layers[node_name] = layers[node.input[0]][starts:ends]
            elif axes == 1:
                layers[node_name] = layers[node.input[0]][:, starts:ends]
            elif axes == 2:
                layers[node_name] = layers[node.input[0]][:, :, starts:ends]
            elif axes == 3:
                layers[node_name] = layers[node.input[0]][:, :, :, starts:ends]
            else:
                raise AttributeError('Not implemented')
        else:
            #raise AttributeError('Not implemented')
            layers[node_name] = layers[node.input[0]][layers[node.input[1]][0]:layers[node.input[2]][0]]


    else:
        logger.debug('Convert inputs to Keras/TF layers if needed.')
        input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)
        layers[node_name] = input_0

        if 'axes' in params:
            if len(params["axes"]) != 1:
                raise NotImplementedError("Multiple axes in Slice is not implemented")
            axes = params["axes"][0]
            ends = params["ends"][0]
            starts = params["starts"][0]
        else:
            starts = ensure_numpy_type(layers[node.input[1]])
            ends = ensure_numpy_type(layers[node.input[2]])
            axes = ensure_numpy_type(layers[node.input[3]])

            for i in range(len(starts)):
                if axes[i] != i:
                    assert AttributeError('Cant slice permuted axes')

        if isinstance(axes, list) or isinstance(axes, np.ndarray):
            if params['change_ordering']:
                raise NotImplementedError("change_ordering for Slice is not implemented")

            def target_layer(x, axes=np.array(axes), starts=starts, ends=ends):
                import tensorflow as tf
                rank = max(axes)
                s = [0 for _ in range(rank+1)]
                e = [0 for _ in range(rank+1)]
                mask = 0xff
                for _s, _e, axis in zip(starts, ends, axes):
                    s[axis] = _s
                    e[axis] = _e
                    mask = mask ^ (0x1 << axis)
                return tf.strided_slice(x, s, e, begin_mask=mask, end_mask=mask)

            lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
            layers[node_name] = lambda_layer(input_0)
            lambda_func[keras_name] = target_layer
        else:
            def target_layer(x, axis=axes, starts=starts, ends=ends):
                import tensorflow as tf
                rank = axis
                s = [0 for _ in range(rank+1)]
                e = [0 for _ in range(rank+1)]
                mask = 0xff
                s[axis] = starts
                e[axis] = ends
                mask = mask ^ (0x1 << axis)
                return tf.strided_slice(x, s, e, begin_mask=mask, end_mask=mask)

            lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
            layers[node_name] = lambda_layer(input_0)
            lambda_func[keras_name] = target_layer


def convert_resize(node, params, layers, lambda_func, node_name, keras_name):
    assert params["coordinate_transformation_mode"] == b"align_corners"
    assert params["mode"] == b"linear"
    new_size = layers[node.input[3]][-2:]
    x = layers[node.input[0]]
    # if not params['change_ordering']:
    def resize_func(x, size=new_size):
        import tensorflow as tf
        is_chw = tf.keras.backend.image_data_format() == "channels_first"
        if is_chw:
            x = tf.transpose(x, [0, 2, 3, 1])
        y = tf.compat.v1.image.resize_bilinear(x, size=size, align_corners=True, half_pixel_centers=False)
        if is_chw:
            y = tf.transpose(y, [0, 3, 1, 2])
        return y
    y = keras.layers.Lambda(resize_func)(x)
    lambda_func[keras_name] = resize_func
    layers[node_name] = y


def convert_global_avg_pool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert GlobalAvgPool layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:global_avg_pool')

    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)

    global_pool = keras.layers.GlobalAveragePooling2D(data_format='channels_first', name=keras_name)
    input_0 = global_pool(input_0)

    logger.debug('Now expand dimensions twice.')
    def _expand_layer(x):
        import tensorflow as tf
        is_chw = tf.keras.backend.image_data_format() == "channels_first"
        if is_chw:
            return x[:, :, None, None]
        else:
            return x[:, None, None, :]
    lambda_layer = keras.layers.Lambda(_expand_layer, name=keras_name + '_EXPAND')
    input_0 = lambda_layer(input_0)  # double expand dims
    layers[node_name] = input_0
    lambda_func[keras_name + '_EXPAND'] = _expand_layer

    
# Register patch
onnx2keras.layers.AVAILABLE_CONVERTERS["Resize"] = convert_resize
onnx2keras.layers.AVAILABLE_CONVERTERS["Slice"] = convert_slice
onnx2keras.layers.AVAILABLE_CONVERTERS["GlobalAveragePool"] = convert_global_avg_pool
