from __future__ import print_function
import sys
import os
import h5py
from pprint import pprint, pformat
import keras
assert keras.__version__.startswith('2.')
import tensorflow as tf
assert tf.__version__.startswith('1.')
import keras.backend.tensorflow_backend as KTF

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    return tf.Session(config=config)


KTF.set_session(get_session())
from keras import backend as K
assert K.common.image_data_format() == 'channels_last'
assert K.common.image_dim_ordering() == 'tf'
from keras.engine import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Dense
from keras.layers.merge import Multiply, Add, Concatenate
from keras.regularizers import l1, l2
'''from keras.initializers import normal, glorot_uniform, average, he_uniform, TruncatedNormal'''
from keras.initializers import normal, glorot_uniform, he_uniform, TruncatedNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.engine.topology import Container
from MyKerasLayers import AcrossChannelLRN, XLayer_b


def build_model(yaml_file, weight_file):
    exp_name, pr_dp, pr_solver, pr_net = yaml_parse(yaml_file)
    opt = SGD(
        lr=pr_solver['lr_base'],
        decay=pr_solver['lr_decay'],
        momentum=pr_solver['momentum'],
        nesterov=pr_solver.get('nesterov', False),
        clipnorm=5,
        clipvalue=1)
    model = tf_build_model_AFCN(
        net_param=pr_net,
        batch_size=pr_dp['train']['batch_size'],
        optimizer=opt,
        base_weight_decay=pr_solver['weight_decay'])
    # for prediction only
    tf_load_weights_pred(model=model,
                         filepath=weight_file)
    # You need to manually set the weights and bias of every layer from the HDF5 file storing the pre-trained weights/bias
    return model


def tf_load_weights_pred(model, filepath, layers_to_init=None):
    # from Keras's load weight. Simply skip useless layers for prediction.
    print('Loading weight...')
    f = h5py.File(filepath, mode='r')
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    print("original_keras_version: {}".format(original_keras_version))
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None
    print("original_backend: {}".format(original_backend))

    if hasattr(model, 'flattened_layers'):
        # support for legacy Sequential/Merge behavior
        print('Legacy mode')
        flattened_layers = model.flattened_layers
    else:
        print('New API')
        flattened_layers = model.layers
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    # print("layer_names: {}".format(layer_names))
    # layer_names: [u'patches', u'conv_1', u'conv_1_act', u'conv_1_norm', u'conv_1_pool', u'conv_2', u'conv_2_act', u'conv_2_norm', u'conv_2_pool', u'conv_3', u'conv_3_act', u'conv_3_norm', u'conv_3_pool', u'flatten_1', u'fc_1', u'fc_1_act', u'fc_1_norm', u'fc_1_cls', u'fc_2', u'fc_1_cls_act', u'fc_2_act', u'fc_1_cls_norm', u'fc_2_norm', u'fc_2_cls', u'fc_3', u'fc_2_cls_act', u'fc_3_act']
    for layer in flattened_layers:
        print("layer in model: {}".format(layer.name))

    weight_value_tuples = []
    k = 0
    ignore_these_layers = [
        'fc_block_1_cls', 'fc_block_1_cls_act', 'fc_block_1_cls_norm',
        'fc_block_2_cls', 'fc_block_2_cls_act', 'fc_block_2_cls_norm',
        'fc_3_count', 'Denses_patch',  # for ZhangCong's model
        'conv_block_1_cls', 'conv_block_2_cls', 'conv_block_3_cls',
        # 'conv_block_1_rgr', 'conv_block_2_rgr', 'conv_block_3_rgr',
        'conv_block_1_cls_act', 'conv_block_2_cls_act', 'conv_block_3_cls_act',
    ]
    for name in layer_names:
        if name in ignore_these_layers:
            # ignore and only ignore the layers in classification branch
            continue
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if len(weight_names):
            weight_values = [g[weight_name] for weight_name in weight_names]
            layer = flattened_layers[k]
            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
            print("{} --> {}".format(name, layer.name))
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '" in the current model) was found to '
                                'correspond to layer ' + name +
                                ' in the save file. '
                                'However the new layer ' + layer.name +
                                ' expects ' + str(len(symbolic_weights)) +
                                ' weights, but the saved weights have ' +
                                str(len(weight_values)) +
                                ' elements.')
            weight_value_tuples += zip(symbolic_weights, weight_values)
        k += 1
    K.batch_set_value(weight_value_tuples)
    f.close()


def yaml_parse(yaml_file, output=True):
    import yaml
    # NOTE: yaml.load can recognize False as False, but can only recognize None as 'None' (str type)
    assert os.path.isfile(yaml_file), "cannot find: {}".format(yaml_file)
    with open(yaml_file, 'r') as f:
        content = yaml.load(f)
    if output:
        '''print(pprint.pformat(content))'''
    exp_name = content['exp_name']
    print('Name of this exp is:')
    print('    {}'.format(exp_name))
    pr_solver = content['solver_params']
    pr_net = content['network_params']
    pr_dp = content['dp_params']
    # _, expName = yaml_file.rsplit('/', 1)
    # expName, _ = expName.rsplit('.', 1)
    # assert expName == exp_name
    # pr_dp = check_params_data_provider_config(pr_dp, output=output)
    # pr_solver = check_params_solver_config(pr_solver, output=output)
    # check_params_net_config(pr_net)
    if output:
        print(pformat(exp_name))
        print('Parameters of Data Provider')
        print(pformat(pr_dp))
        print('^^^^^^^^^ ^^^^^^^^^')
        print('Parameters of Solver')
        print(pformat(pr_solver))
        print('^^^^^^^^^ ^^^^^^^^^')
        print('Parameters of NN')
        print(pformat(pr_net))
        print('^^^^^^^^^ ^^^^^^^^^')
        # time.sleep(15)
    return exp_name, pr_dp, pr_solver, pr_net


def load_weights(filename):
    # use weight_dict[layer_name][weight_name] to access the ndarray
    # e.g. weight_dict['conv_1']['conv_1/kernel:0'] and weight_dict['conv_1']['conv_1/bias:0']
    assert os.path.isfile(filename)
    weight_dict = dict()
    with h5py.File(filename, 'r') as f:
        if 'keras_version' in f.attrs:
            original_keras_version = f.attrs['keras_version'].decode('utf8')
        else:
            original_keras_version = '1'
        print("original_keras_version: {}".format(original_keras_version))
        if 'backend' in f.attrs:
            original_backend = f.attrs['backend'].decode('utf8')
        else:
            original_backend = None
        print("original_backend: {}".format(original_backend))
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        for layer in layer_names:
            weight_names = [n.decode('utf8') for n in f[layer].attrs['weight_names']]
            if len(weight_names):
                weight_dict[layer] = dict()
                for weight_name in weight_names:
                    weight_dict[layer][weight_name] = f[layer][weight_name].value
    return weight_dict


def _screen_print(name, net_param, verbosity=True):
    if verbosity:
        print('========= ========= ========= =========')
        print('Layer {}'.format(name))
        print(pformat(net_param))


def conv_block(input_tensor, conv_block_params, base_weight_decay=0.0005):
    # pooling comes first
    # if using LRN, then do activation before LRN
    # elif using BatchNorm, then do activation after batch norm
    conv_layer_params = conv_block_params['conv']
    activation_layer_params = conv_block_params.get('act', {type: 'linear'})
    pool_layer_params = conv_block_params.get('pool', None)
    norm_layer_params = conv_block_params.get('norm', None)
    name_conv = conv_block_params['name']
    name_pool = '{}_pool'.format(name_conv)
    name_act = '{}_act'.format(name_conv)
    name_norm = '{}_norm'.format(name_conv)
    output_name = None
    if 'std_gaussian' in conv_layer_params:
        init_func = TruncatedNormal(
            mean=0.,
            stddev=conv_layer_params['std_gaussian']
        )
    elif 'init' in conv_layer_params:
        init_func = conv_layer_params['init']
        print('init function for layer-{}: {}'.format(name_conv, init_func))
    else:
        print('Explicitly specify init please.')
        sys.exit(1)
    assert conv_layer_params.get('activation', None) in [None, 'linear']
    _screen_print(name_conv, conv_layer_params)
    assert conv_layer_params['type'] == 'conv2d', "you are trying to use: {}".format(
        conv_layer_params['type'])
    x = Conv2D(
        data_format='channels_last',
        trainable=conv_layer_params.get('trainable', True),
        filters=conv_layer_params['nb_out_feat_map'],
        # kernel_size: An integer or tuple/list of 2 integers, specifying (filter_height, filter_width)
        kernel_size=(conv_layer_params['filter_h'], conv_layer_params['filter_w']),
        strides=(conv_layer_params.get('stride_h', 1),
                 conv_layer_params.get('stride_w', 1)),
        kernel_initializer=init_func,
        padding=conv_layer_params['border_mode'],
        kernel_regularizer=l2(base_weight_decay * conv_layer_params['decay_mult']),
        # use_bias=conv_layer_params.get('bias', True),
        use_bias=False if (norm_layer_params is not None and norm_layer_params['type'].lower() in [
            'bn', 'batchnorm', 'batchnormalization']) else conv_layer_params.get('bias', True),
        activation=None,
        name=name_conv
    )(input_tensor)

    if pool_layer_params is not None:
        _screen_print(name_pool, pool_layer_params)
        x = MaxPooling2D(
            pool_size=(pool_layer_params['size_h'],
                       pool_layer_params['size_w']),
            strides=(pool_layer_params['stride_h'],
                     pool_layer_params['stride_w']),
            padding=pool_layer_params['border_mode'],
            name=name_pool)(x)

    if norm_layer_params is not None:
        if norm_layer_params['type'].lower() == 'lrn':
            _screen_print(name_act, activation_layer_params)
            if activation_layer_params['type'].lower() not in ['leakyrelu']:
                x = Activation(activation_layer_params['type'], name=name_act)(x)
            else:
                if activation_layer_params['type'].lower() == 'leakyrelu':
                    print("use LeakyReLU")
                    x = LeakyReLU(alpha=activation_layer_params['alpha'], name=name_act)(x)
                else:
                    print("Unknown activation.")
                    sys.exit(1)
            #
            _screen_print(name_norm, norm_layer_params)
            x = AcrossChannelLRN(
                local_size=norm_layer_params['local_size'],
                alpha=norm_layer_params['alpha'],
                beta=norm_layer_params['beta'],
                k=norm_layer_params['k'],
                name=name_norm)(x)
            output_name = name_norm
        elif norm_layer_params['type'].lower() in ['bn', 'batchnorm', 'batchnormalization']:
            _screen_print(name_norm, norm_layer_params)
            print('add Batch Normalization layer')
            assert norm_layer_params['axis'] == 3
            x = BatchNormalization(
                axis=norm_layer_params['axis'],
                momentum=norm_layer_params['momentum'],  # default setting: 0.99
                epsilon=1e-3,  # default setting: 1e-3
                name=name_norm)(x)
            #
            _screen_print(name_act, activation_layer_params)
            if activation_layer_params['type'].lower() not in ['leakyrelu']:
                x = Activation(activation_layer_params['type'], name=name_act)(x)
            else:
                if activation_layer_params['type'].lower() == 'leakyrelu':
                    print("use LeakyReLU")
                    x = LeakyReLU(alpha=activation_layer_params['alpha'], name=name_act)(x)
                else:
                    print("Unknown activation.")
                    sys.exit(1)
            output_name = name_act
        elif norm_layer_params['type'] == 'dropout':
            _screen_print(name_act, activation_layer_params)
            if activation_layer_params['type'].lower() not in ['leakyrelu']:
                x = Activation(activation_layer_params['type'], name=name_act)(x)
            else:
                if activation_layer_params['type'].lower() == 'leakyrelu':
                    print("use LeakyReLU")
                    x = LeakyReLU(alpha=activation_layer_params['alpha'], name=name_act)(x)
                else:
                    print("Unknown activation.")
                    sys.exit(1)
            _screen_print(name_norm, norm_layer_params)
            x = Dropout(norm_layer_params['drop_ratio'],
                        name=name_norm)(x)
            output_name = name_norm
        else:
            print('Unrecognized normalization type')
            sys.exit(1)
    else:
        _screen_print(name_act, activation_layer_params)
        if activation_layer_params['type'].lower() not in ['leakyrelu']:
            x = Activation(activation_layer_params['type'], name=name_act)(x)
        else:
            if activation_layer_params['type'].lower() == 'leakyrelu':
                print("use LeakyReLU")
                x = LeakyReLU(alpha=activation_layer_params['alpha'], name=name_act)(x)
            else:
                print("Unknown activation.")
                sys.exit(1)
        output_name = name_act
    assert output_name
    return x, output_name


def adapt_conv_block(input_tensor, input_aux_param,
                     conv_block_params, base_weight_decay=0.0005):
    # if using LRN, then do activation before LRN
    # elif using BatchNorm, then do activation after batch norm
    conv_layer_params = conv_block_params['conv']
    activation_layer_params = conv_block_params.get('act', {type: 'linear'})
    pool_layer_params = conv_block_params.get('pool', None)
    norm_layer_params = conv_block_params.get('norm', None)
    name_conv = conv_block_params['name']
    name_pool = '{}_pool'.format(name_conv)
    name_act = '{}_act'.format(name_conv)
    name_norm = '{}_norm'.format(name_conv)
    output_name = None
    # ********* ********* ********* *********
    # X branch
    # ********* ********* ********* *********
    idx = 1
    curr_layer_name = '{}_FMN_fc_{}'.format(name_conv, idx)
    print('^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^')
    print('Layer {}'.format(curr_layer_name))
    print(pformat(conv_layer_params['FMN']['fc_{}'.format(idx)]))
    if 'std_gaussian' in conv_layer_params['FMN']['fc_{}'.format(idx)]:
        print("Use simple normal init")
        init_func = TruncatedNormal(
            mean=0.,
            stddev=conv_layer_params['FMN']['fc_{}'.format(idx)]['std_gaussian']
        )
    elif 'init' in conv_layer_params['FMN']['fc_{}'.format(idx)]:
        init_func = conv_layer_params['FMN']['fc_{}'.format(idx)]['init']
        print('init function for layer-{}: {}'.format('fc_{}'.format(idx), init_func))
    else:
        print('Explicitly specify init please.')
        sys.exit(1)
    if conv_layer_params['FMN']['fc_{}'.format(idx)]['decay_mult'] > 1:
        print('Use stronger weight decay ({}) for layer: {}'.format(conv_layer_params[
              'FMN']['fc_{}'.format(idx)]['decay_mult'], 'fc_{}'.format(idx)))
    filter_params = Dense(
        units=conv_layer_params['FMN']['fc_{}'.format(idx)]['nb_out'],
        use_bias=conv_layer_params['FMN']['fc_{}'.format(idx)].get('bias', True),
        trainable=conv_layer_params['FMN'][
            'fc_{}'.format(idx)].get('trainable', True),
        kernel_initializer=init_func,
        activation=conv_layer_params['FMN']['fc_{}'.format(idx)]['activation'],
        kernel_regularizer=l2(
            base_weight_decay * conv_layer_params['FMN']['fc_{}'.format(idx)]['decay_mult']),
        name=curr_layer_name
    )(input_aux_param)

    if 'norm' in conv_layer_params['FMN']['fc_{}'.format(idx)] and conv_layer_params['FMN']['fc_{}'.format(idx)]['norm'] == 'dropout':
        curr_layer_name = '{}_FMN_fc_{}_norm'.format(name_conv, idx)
        filter_params = Dropout(0.5, name=curr_layer_name)(filter_params)

    for idx in range(2, 5):
        if conv_layer_params['FMN'].get('fc_{}'.format(idx), None):
            curr_layer_name = '{}_FMN_fc_{}'.format(name_conv, idx)
            print('^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^')
            print('Layer {}'.format(curr_layer_name))
            print(pformat(conv_layer_params['FMN']['fc_{}'.format(idx)]))
            if 'std_gaussian' in conv_layer_params['FMN']['fc_{}'.format(idx)]:
                print("Use simple normal init")
                # init_func = lambda shape, name: normal(shape, scale=conv_layer_params['FMN'][
                #                                        'fc_{}'.format(idx)]['std_gaussian'], name=name)
                init_func = TruncatedNormal(
                    mean=0.,
                    stddev=conv_layer_params['FMN']['fc_{}'.format(idx)]['std_gaussian']
                )
            elif 'init' in conv_layer_params['FMN']['fc_{}'.format(idx)]:
                # print("Please use Normal init.")
                # sys.exit(1)
                init_func = conv_layer_params['FMN']['fc_{}'.format(idx)][
                    'init']
                print(
                    'init function for layer-{}: {}'.format('fc_{}'.format(idx), init_func))
            else:
                print('Explicitly specify init please.')
                sys.exit(1)
            if conv_layer_params['FMN']['fc_{}'.format(idx)]['decay_mult'] > 1:
                print('Use stronger weight decay ({}) for layer: {}'.format(conv_layer_params[
                      'FMN']['fc_{}'.format(idx)]['decay_mult'], 'fc_{}'.format(idx)))
            filter_params = Dense(
                units=conv_layer_params['FMN']['fc_{}'.format(idx)]['nb_out'],
                use_bias=conv_layer_params['FMN'][
                    'fc_{}'.format(idx)].get('bias', True),
                trainable=conv_layer_params['FMN'][
                    'fc_{}'.format(idx)].get('trainable', True),
                kernel_initializer=init_func,
                activation=conv_layer_params['FMN'][
                    'fc_{}'.format(idx)]['activation'],
                kernel_regularizer=l2(
                    base_weight_decay * conv_layer_params['FMN']['fc_{}'.format(idx)]['decay_mult']),
                name=curr_layer_name
            )(filter_params)

            if 'norm' in conv_layer_params['FMN']['fc_{}'.format(idx)] and conv_layer_params['FMN']['fc_{}'.format(idx)]['norm'] == 'dropout':
                curr_layer_name = '{}_FMN_fc_{}_norm'.format(name_conv, idx)
                filter_params = Dropout(0.5, name=curr_layer_name)(filter_params)
        else:
            break

    filter_net = Model(
        inputs=input_aux_param,
        outputs=filter_params,
        name='{}_FMN'.format(name_conv)
    )

    _screen_print(name_conv, conv_layer_params)
    assert conv_layer_params['type'].lower().startswith(
        'xlayer_b'), "you are trying to use: {}".format(conv_layer_params['type'])
    x = XLayer_b(
        weight_param_net=filter_net,
        data_format='channels_last',
        filters=conv_layer_params['nb_out_feat_map'],
        kernel_size=(conv_layer_params['filter_h'], conv_layer_params['filter_w']),
        strides=(conv_layer_params.get('stride_h', 1),
                 conv_layer_params.get('stride_w', 1)),
        use_bias=False if (norm_layer_params is not None and norm_layer_params['type'].lower() in [
            'bn', 'batchnorm', 'batchnormalization']) else conv_layer_params.get('bias', True),
        padding=conv_layer_params['border_mode'],
        activation='linear',
        kernel_regularizer=l2(base_weight_decay * conv_layer_params['decay_mult']),
        name=name_conv,
    )(input_tensor)

    # below are exactly the same as conv_block
    if pool_layer_params is not None:
        _screen_print(name_pool, pool_layer_params)
        x = MaxPooling2D(
            pool_size=(pool_layer_params['size_h'],
                       pool_layer_params['size_w']),
            strides=(pool_layer_params['stride_h'],
                     pool_layer_params['stride_w']),
            padding=pool_layer_params['border_mode'],
            name=name_pool,
        )(x)

    if norm_layer_params is not None:
        if norm_layer_params['type'].lower() == 'lrn':
            _screen_print(name_act, activation_layer_params)
            if activation_layer_params['type'].lower() not in ['leakyrelu']:
                x = Activation(activation_layer_params['type'], name=name_act)(x)
            else:
                if activation_layer_params['type'].lower() == 'leakyrelu':
                    print("use LeakyReLU")
                    x = LeakyReLU(alpha=activation_layer_params['alpha'], name=name_act)(x)
                else:
                    print("Unknown activation.")
                    sys.exit(1)
            #
            _screen_print(name_norm, norm_layer_params)
            x = AcrossChannelLRN(
                local_size=norm_layer_params['local_size'],
                alpha=norm_layer_params['alpha'],
                beta=norm_layer_params['beta'],
                k=norm_layer_params['k'],
                name=name_norm)(x)
            output_name = name_norm
        elif norm_layer_params['type'].lower() in ['bn', 'batchnorm', 'batchnormalization']:
            _screen_print(name_norm, norm_layer_params)
            print('add Batch Normalization layer')
            assert norm_layer_params['axis'] == 3
            x = BatchNormalization(
                axis=norm_layer_params['axis'],
                momentum=norm_layer_params['momentum'],  # default setting: 0.99
                epsilon=1e-3,  # default setting: 1e-3
                name=name_norm)(x)
            #
            _screen_print(name_act, activation_layer_params)
            if activation_layer_params['type'].lower() not in ['leakyrelu']:
                x = Activation(activation_layer_params['type'], name=name_act)(x)
            else:
                if activation_layer_params['type'].lower() == 'leakyrelu':
                    print("use LeakyReLU")
                    x = LeakyReLU(alpha=activation_layer_params['alpha'], name=name_act)(x)
                else:
                    print("Unknown activation.")
                    sys.exit(1)
            output_name = name_act
        elif norm_layer_params['type'] == 'dropout':
            _screen_print(activation_layer_params[
                          'name'], activation_layer_params)
            if activation_layer_params['type'].lower() not in ['leakyrelu']:
                x = Activation(activation_layer_params['type'], name=name_act)(x)
            else:
                if activation_layer_params['type'].lower() == 'leakyrelu':
                    print("use LeakyReLU")
                    x = LeakyReLU(alpha=activation_layer_params['alpha'], name=name_act)(x)
                else:
                    print("Unknown activation.")
                    sys.exit(1)
            _screen_print(name_norm, norm_layer_params)
            x = Dropout(norm_layer_params['drop_ratio'],
                        name=name_norm)(x)
            output_name = name_norm
        else:
            print('Unrecognized normalization type')
            sys.exit(1)
    else:
        _screen_print(name_act, activation_layer_params)
        if activation_layer_params['type'].lower() not in ['leakyrelu']:
            x = Activation(activation_layer_params['type'], name=name_act)(x)
        else:
            if activation_layer_params['type'].lower() == 'leakyrelu':
                print("use LeakyReLU")
                x = LeakyReLU(alpha=activation_layer_params['alpha'], name=name_act)(x)
            else:
                print("Unknown activation.")
                sys.exit(1)
        output_name = name_act
    assert output_name
    return x, output_name


def tf_build_model_AFCN(net_param, batch_size, optimizer,
                        base_weight_decay=0.0005):
    print('Using tf_build_model_AFCN')
    if net_param['arch_type'].get('w_regularization_type', 'l2') == 'l2':
        print('use l2 w_regularization')
        # w_reg = l2
    elif net_param['arch_type'].get('w_regularization_type', 'l2') == 'l1':
        print('use l1 w_regularization')
        print("Do NOT support yet.")
        sys.exit(1)
        # w_reg = l1
    else:
        print('Only support l1 and l2 w_regularization so far')
        sys.exit(1)

    input_shape = (batch_size,
                   net_param['patches']['dim_h'],
                   net_param['patches']['dim_w'],
                   net_param['patches']['dim_c'])
    input_patches = Input(batch_shape=input_shape, name='patches')
    if 'output_masks' in net_param:
        output_masks = Input(
            batch_shape=(batch_size,
                         net_param['output_masks']['dim_h'],
                         net_param['output_masks']['dim_w'], 1), name='output_masks')
    input_aux_param = Input(batch_shape=(batch_size, net_param['aux_params']['dim']),
                            name='aux_params')

    if net_param['conv_block_1']['type'] == 'adaptive_conv_block':
        conv_part, _ = adapt_conv_block(
            input_tensor=input_patches,
            input_aux_param=input_aux_param,
            conv_block_params=net_param['conv_block_1'],
            base_weight_decay=base_weight_decay,
        )
    elif net_param['conv_block_1']['type'] == 'conv_block':
        conv_part, _ = conv_block(
            input_tensor=input_patches,
            conv_block_params=net_param['conv_block_{}'.format(1)],
            base_weight_decay=base_weight_decay,
        )

    for idx in range(2, 20, 1):
        if net_param.get('conv_block_{}'.format(idx), None):
            if net_param['conv_block_{}'.format(idx)]['type'] == 'adaptive_conv_block':
                conv_part, rgr_output = adapt_conv_block(
                    input_tensor=conv_part,
                    input_aux_param=input_aux_param,
                    conv_block_params=net_param['conv_block_{}'.format(idx)],
                    base_weight_decay=base_weight_decay,
                )
            elif net_param['conv_block_{}'.format(idx)]['type'] == 'conv_block':
                conv_part, rgr_output = conv_block(
                    input_tensor=conv_part,
                    conv_block_params=net_param['conv_block_{}'.format(idx)],
                    base_weight_decay=base_weight_decay,
                )
            else:
                print("only support [conv_block, adaptive_conv_block]")
                sys.exit(1)
        else:
            break

    if 'output_masks' in net_param:
        rgr_output = 'den_map_roi'
        output = Multiply(name=rgr_output)([conv_part, output_masks])
        model = Model(inputs=[input_patches, input_aux_param, output_masks],
                      outputs=output,
                      name=net_param['net_name'])
    else:
        model = Model(inputs=[input_patches, input_aux_param],
                      outputs=conv_part,
                      name=net_param['net_name'])
    print('Layer name of regression output: {}'.format(rgr_output))
    print('Compiling ...')
    model.compile(
        optimizer=optimizer,
        loss={rgr_output: net_param['loss_rgr']},
        loss_weights={rgr_output: net_param.get(
                      'loss_rgr_weight', 1)}
    )
    return model
