# #############################################################################
# inference.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################
import random
import numpy as np
import cv2
import scipy
from scipy.fftpack import next_fast_len

import tensorflow as tf

import keras
from keras import Model, regularizers
from keras.utils.layer_utils import count_params
from keras.layers import Input, BatchNormalization, UpSampling2D, Concatenate, ZeroPadding2D, Lambda, Cropping2D, GroupNormalization, GlobalAveragePooling2D, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.losses import Loss
from keras import optimizers

import tensorflow_model_optimization as tfmot
from keras_unet_collection import models as M_unet



#################################################################################################################
####################################### Camera inversion layers #################################################
#################################################################################################################

def to_channel_last(x):
    """from NCHW to NHWC format

    Args:
        x (tf tensor): input in NCHW format

    Returns:
        tf tensor: output in NHWC format
    """
    return tf.keras.layers.Permute([2, 3, 1])(x)
    
    
def to_channel_first(x):
    """from NHWC to NCHW format

    Args:
        x (tf tensor): input in NHWC format

    Returns:
        tf tensor: output in NCHW format
    """
    return tf.keras.layers.Permute([3, 1, 2])(x)

############################# Separable ##############################

def get_toeplitz_init(target_shape, slope, is_left=False, seed=1):
    # consider that 
    first_d, second_d = target_shape
    # is_left = first_d >= second_d

    if is_left:
        # cv2 : in (width, height) format
        resized_dims = (int(first_d * slope), first_d)
        np.random.seed(seed)
        arr = np.random.rand(first_d)

        circulant_M = scipy.linalg.circulant(arr)

        resized_M = cv2.resize(circulant_M, resized_dims, interpolation= cv2.INTER_LINEAR)
        random.seed(seed)
        begin_index = random.choice(range(resized_M.shape[1] - second_d))

        cropped_M = resized_M[:, begin_index : begin_index + second_d]

    else : 
        # to get diff matrix for left and right
        seed += 1
        resized_dims = (second_d, int(second_d * slope))
        np.random.seed(seed)
        arr = np.random.rand(second_d)

        circulant_M = scipy.linalg.circulant(arr)

        resized_M = cv2.resize(circulant_M, resized_dims, interpolation= cv2.INTER_LINEAR)
        random.seed(seed)
        begin_index = random.choice(range(resized_M.shape[0] - first_d))

        cropped_M = resized_M[begin_index : begin_index + first_d, :]

    return cropped_M


class SeparableLayer(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer, tfmot.clustering.keras.ClusterableLayer):
    """Layer used for the trainable inversion in FlatNet for the separable case

    Args:
        W1_init (np array): initial value for W1
        W2_init (np array): initial value for W2
        name (str, optional): name of the layer. Defaults to 'separable_layer'.
    """
    def __init__(self, W1_init, W2_init, name='separable_layer'):
        super(SeparableLayer, self).__init__(name=name)
        self.W1 = tf.Variable(W1_init, name='camera_inversion_W1')
        self.W2 = tf.Variable(W2_init, name='camera_inversion_W2')
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)



    def build(self, input_shape):
        self.in_shape = input_shape
        b, h, w, c = self.in_shape
        # print('W1 shape:', self.W1.shape)
        # print('W2 shape:', self.W2.shape)
        # print('input shape:', self.in_shape)
        assert h == self.W1.shape[0], f"W1 width must be equal to the input height, got {self.W1.shape[0]} and {h}"
        assert w == self.W2.shape[0], f"W1 height must be equal to the input width, got , got {self.W2.shape[0]} and {w}"


    def get_config(self):
        config = super().get_config()

        config.update({
            "W1": self.W1,
            "W2": self.W2
        })
        return config
    

    def call(self, x):
        #In NCHW format: tf.matmul inner-most 2 dimensions
        x = to_channel_first(x)
        x = tf.matmul(self.W1, x, transpose_a=True)
        x = tf.matmul(x, self.W2)
        x = to_channel_last(x)

        return self.activation(x)
    

    def get_list_weights(self):
        return [self.W1, self.W2]

    def set_list_weights(self, list_weights):
        self.W1 = list_weights[0]
        self.W2 = list_weights[1]

    def get_prunable_weights(self):
        return [self.W1, self.W2]
    
    def get_clusterable_weights(self):
        return [('W1', self.W1), ('W2', self.W2)]



    

############################## non-separable ##############################



def get_wiener_matrix(psf, gamma: int = 20000):
    """get Wiener matrix of PSF

    Args:
        psf (numpy array): point-spread-function matrix, shape (H, W, C)
        gamma (int, optional): regularization parameter. Defaults to 20000.

    Returns:
        numpy array: wiener filter of psf
    """

    H = np.fft.rfft2(psf, axes=(0, 1))
    H_conj = np.conj(H)

    H_absq = np.abs(H)**2

    res = np.fft.irfft2(H_conj / (gamma + H_absq).astype(np.complex64), axes=(0, 1), s=psf.shape[:2])

    return res.astype(np.float32)





class FTLayer(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer, tfmot.clustering.keras.ClusterableLayer):
    """Layer used for the trainable inversion in FlatNet for the non-separable case

    Args:
        psf (numpy array): point-spread-function matrix, shape (H, W, C)
        activation (str, optional): activation function. Defaults to 'linear'.
        gamma (int, optional): regularization parameter. Defaults to 20000.
        pad (bool, optional): whether to pad the input or not to do a valid convolution in frequency domain. Defaults to False.
        name (str, optional): name of the layer. Defaults to 'non_separable_layer'.
    """
    def __init__(self, 
                 psf, 
                 activation='linear', 
                 gamma=20000, 
                 pad=False, 
                 name='non_separable_layer', 
                 **kwargs):
        
        super(FTLayer, self).__init__(name=name, **kwargs)
        self.psf = psf
        self.pad = pad
        self.activation = tf.keras.activations.get(activation)
        self.gamma = gamma

        self.psf_shape = psf.shape

        wiener_crop = tf.convert_to_tensor(get_wiener_matrix(psf, gamma=self.gamma))
        wiener_crop = tf.transpose(wiener_crop, (2, 0, 1))
        
        self.W = tf.Variable(wiener_crop, name='camera_inversion_W')

        self.normalizer = tf.Variable([[[[1 / 0.0008]]]], shape=(1, 1, 1, 1), name='camera_inversion_normalizer')



    def build(self, input_shape):
        channel = input_shape[3]
        
        psf_shape = np.asarray(self.psf_shape[:2])
        in_shape = np.asarray(input_shape[1:3])
        
        assert np.all(psf_shape >= in_shape), 'PSF shape must be greater than input shape'

        target_shape = 2 * in_shape - 1 if self.pad else psf_shape

        self._start_idx_input, self._end_idx_input = self._get_pad_idx(img_shape=in_shape, target_shape=target_shape, channel=channel)
        # to pad to efficient computation size
        self._start_idx_psf, self._end_idx_psf = self._get_pad_idx(img_shape=psf_shape, target_shape=target_shape, channel=channel)
        

        
    def _get_pad_idx(self, img_shape, target_shape, channel):
        padded_shape = np.asarray(target_shape)

        padded_shape = np.array([next_fast_len(i) for i in padded_shape])
        # print('padded shape', padded_shape)
        padded_shape = list(np.r_[padded_shape, channel])

        start_idx = (padded_shape[0 : 2] - img_shape) // 2

        end_idx = start_idx + (padded_shape[0 : 2] - img_shape) % 2
        return start_idx, end_idx
        
    
    def get_config(self):
        config = super().get_config()

        config.update({
            "psf": self.psf,
            "activation": self.activation,
            "pad": self.pad,
            "gamma": self.gamma
        })
        return config
      

    def _to_ft(self, w):
        w = tf.pad(w, ((0,0),
                       (self._start_idx_psf[0], self._end_idx_psf[0]),
                       (self._start_idx_psf[1], self._end_idx_psf[1])), "CONSTANT")
        

        return tf.signal.rfft2d(w)


    def call(self, x):        
        x = tf.pad(x, ((0,0),
                       (self._start_idx_input[0], self._end_idx_input[0]),
                       (self._start_idx_input[1], self._end_idx_input[1]), (0,0)), "CONSTANT")
         
        x = to_channel_first(x)
        
        W = self._to_ft(self.W)

        mult = tf.signal.rfft2d(x) * W

        x = tf.signal.ifftshift(tf.signal.irfft2d(mult),
                                axes=(-2, -1))
        
        x = Cropping2D(cropping=((self._start_idx_input[0], self._end_idx_input[0]),
                                     (self._start_idx_input[1], self._end_idx_input[1])),
                                     data_format='channels_first')(x)

        x = x * self.normalizer

        x = to_channel_last(x)

        return self.activation(x)
    
    def get_list_weights(self):
        return [self.W, self.normalizer]
    
    def set_list_weights(self, list_weights):
        self.W = list_weights[0]
        self.normalizer = list_weights[1]

    def get_prunable_weights(self):
        return [self.W]
    
    def get_clusterable_weights(self):
        return [('W', self.W)]




#################################################################################################################
######################################### Perceptual Model ######################################################
#################################################################################################################


######################################### Basic U-Net ##########################################################

def conv_block(x, 
               filters, 
               kernel_size, 
               strides=1, 
               bn_eps=1e-3, 
               l1_factor=0, 
               l2_factor=0, 
               padding='same'):
    '''Convolution block : 2D convolution --> batchnormalization --> relu

    Args:
        x (input): input
        filters (int): number of filters
        kernel_size (int): kernel size
        strides (int, optional): strides number. Defaults to 1.
        l1_factor (int, optional): l1 regulation factor. Defaults to 0.
        l2_factor (int, optional): l2 regulation factor. Defaults to 0.
        padding (str, optional): padding type. Defaults to 'same'.
    Returns:
        output: input passed through convolution block
    '''
    
    x = Conv2D(filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer='glorot_uniform'
                ,kernel_regularizer=regularizers.L1L2(l1=l1_factor, l2=l2_factor)
                )(x)

    # axis=1 for NCHW
    x = BatchNormalization(epsilon=bn_eps)(x)
    x = Activation('relu')(x)
    return x


def stack_encoder(input, filters, kernel_sizes=[3, 3], bn_eps=1e-3, maxpool=True, intermediate_nodes=False):
    if not maxpool:
        padding = ['valid', 'same']
        strides = [2, 1]
    else:
        padding = ['same', 'same']
        strides = [1, 1]
    x = conv_block(input, filters, kernel_sizes[0], bn_eps=bn_eps, strides=strides[0], padding=padding[0])
    x = conv_block(x, filters, kernel_sizes[1], bn_eps=bn_eps, strides=strides[1], padding=padding[1])
    down_tensor = x
    if maxpool:
        x_small = MaxPooling2D(pool_size=2, strides=2)(x)
        if intermediate_nodes:
            down_tensor = conv_block(x_small, filters, kernel_size=1, bn_eps=bn_eps, strides=1, padding='same')
        return x_small, down_tensor
    else:
        if intermediate_nodes:
            input = conv_block(input, filters, kernel_size=1, bn_eps=bn_eps, strides=1, padding='same')
        return down_tensor, input

    


def stack_decoder(x, filters, down_tensor, kernel_size=3, bilinear=True, bn_eps=1e-3, num_conv=2):
    height, width = down_tensor.shape[2:]

    if bilinear:
        #  Exact upsampling
        # x = to_channel_last(x)
        # x = Resizing(height, width,interpolation='bilinear')(x)
        # x = to_channel_first(x)
        x = UpSampling2D(size=2, interpolation="bilinear")(x)
        x = ZeroPadding2D(((0, down_tensor.shape[1] - x.shape[1]), (0, down_tensor.shape[2] - x.shape[2])))(x)

    else:
        raise NotImplementedError
        # Transposed convolution 
        # x = Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
        # y = tf.keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)

    # Concatenate in NHWC
    x = Concatenate(axis=-1)([x, down_tensor])
    # decode : TODO : normally only 2
    for i in range(num_conv):
        x = conv_block(x, filters, kernel_size, bn_eps=bn_eps)

    return x


def resize_input(input, factor):
    while input % factor != 0:
        input += 1
    return input


def u_net(input, enc_filters, maxpool=True, intermediate_nodes=False, first_kernel_size=3,
          last_conv_filter=None, num_dec_conv=2, bn_eps=1e-3, out_shape=None, output_activation='tanh',
          depth_space=False):
    
    x = input
    if depth_space:
        factor = 2
        # height, width = input.shape[1:3]
        # new_height = resize_input(height, factor=factor)
        # new_width = resize_input(width, factor=factor)
        # print('new_height', new_height, 'new_width', new_width)

        # x = tf.keras.layers.Resizing(new_height, new_width)(input)
        x = tf.nn.space_to_depth(x, factor)


    ### down: encoder ###

    down_tensors = []

    if not maxpool:
        x = conv_block(x, enc_filters[0], kernel_size=3, bn_eps=bn_eps, strides=1)
        x = conv_block(x, enc_filters[0], kernel_size=3, bn_eps=bn_eps, strides=1)

    kernel_sizes = [[3,3]] * len(enc_filters)
    if not first_kernel_size:
        first_kernel_size = 3
    kernel_sizes[0] = [first_kernel_size,3]

    
    for i in range(len(enc_filters)):
        x, down_tensor = stack_encoder(x, enc_filters[i], kernel_sizes=kernel_sizes[i], bn_eps=bn_eps, maxpool=maxpool, intermediate_nodes=intermediate_nodes)
        down_tensors.append(down_tensor)
    
    ### Center ###
    x = conv_block(x, filters=enc_filters[-1], kernel_size=3, bn_eps=bn_eps) 

    ### up: decoder ###
    down_tensors = down_tensors[::-1]
    dec_filters = enc_filters[::-1]
    dec_filters = dec_filters[1:] +[dec_filters[-1]]

    for dec_filter, down_tensor in zip(dec_filters, down_tensors):
        x = stack_decoder(x, dec_filter, down_tensor, kernel_size=3, bn_eps=bn_eps, num_conv=num_dec_conv)

    if last_conv_filter:
        x = conv_block(x, last_conv_filter, kernel_size=3)
    
    
    ### "classifier" ###
    num_outputs = 3 if input.shape[1] != 1 else 1
    if depth_space:
        num_outputs *= 2**2
    x = Conv2D(filters=num_outputs, kernel_size=1, use_bias=True, padding='same', activation=output_activation)(x)

    if depth_space:
        x = tf.nn.depth_to_space(x, 2)

    if out_shape:
        # works, but not quantized
        size = out_shape[0:2]
        x = Lambda(lambda x: tf.image.resize(x, size=size, method=tf.image.ResizeMethod.BILINEAR))(x)

    return x



####################################### U-Net variation models ##################################################

MODELS = dict(
            unet_2d = M_unet.unet_2d,
            vnet_2d = M_unet.vnet_2d,
            unet_plus_2d = M_unet.unet_plus_2d,
            r2_unet_2d = M_unet.r2_unet_2d,
            att_unet_2d = M_unet.att_unet_2d,
            resunet_a_2d = M_unet.resunet_a_2d,
            u2net_2d = M_unet.u2net_2d,
            unet_3plus_2d = M_unet.unet_3plus_2d,
            transunet_2d = M_unet.transunet_2d,
            swin_unet_2d = M_unet.swin_unet_2d)


def resize_input(dim, unet_depth):
    while dim % 2**(unet_depth + 1) != 0:
        dim += 1
    return dim


def experimental_models(model_name, input, out_shape, model_args):
    model_args = dict(model_args)

    output_activation = model_args.pop('output_activation')
    model_args['output_activation'] = None

    _, height, width, channels = input.shape

    if 'filter_num' in model_args:
        unet_depth = len(model_args['filter_num'])
        

    elif 'filter_num_down' in model_args:
        unet_depth = len(model_args['filter_num_down'])

    elif 'depth' in model_args:
        unet_depth = model_args['depth']
        
    new_height = resize_input(height, unet_depth)
    new_width = resize_input(width, unet_depth)

    if model_name in ['transunet_2d', 'swin_unet_2d', 'unet_3plus_2d']:
        new_height = max(resize_input(height, unet_depth), resize_input(width, unet_depth))
        new_width = new_height

    x = tf.keras.layers.Resizing(new_height, new_width)(input)
    
    model_args['n_labels'] = channels
    model_args['input_size'] = (new_height, new_width, channels)
    model_args['name'] = model_name
    
    gen_model = MODELS[model_name](**model_args)
    gen_model.summary()

    x = gen_model(x)

    print('output shape: ')
    if output_activation:
        x = Activation(output_activation)(x)

    x = tf.keras.layers.Resizing(height=out_shape[0], width=out_shape[1])(x)

    return x







#################################################################################################################
######################################## Reconstruction Model ###################################################
#################################################################################################################


class UnetModel(keras.Sequential):
    """Reconstruction model for deep learning approach
    """
    def __init__(self, 
                 input_shape, 
                 output_shape, 
                 psf=None,
                 phi_l=None,
                 phi_r=None,
                 perceptual_args=None,
                 camera_inversion_args=None, 
                 model_weights_path=None, 
                 name='reconstruction_model'):
        
        super().__init__(name=name)

        self.in_shape = input_shape

        input = Input(shape=input_shape, name='input', dtype='float32')
        x = input
        # camera inversion layer #
        camera_inversion_layer = None
        if camera_inversion_args:
            # Separable dataset
            if camera_inversion_args['type'] == 'separable':
                if phi_l is None or phi_r is None:
                    print('phi_l and phi_r not provided, using random init')
                    target_shape = camera_inversion_args['target_shape']
                    slope = camera_inversion_args['slope']
                    phi_l = get_toeplitz_init(target_shape, slope, is_left=True, seed=1)
                    phi_r = get_toeplitz_init(target_shape, slope, is_left=False, seed=1)

                camera_inversion_layer = SeparableLayer(phi_l, phi_r)

            # Non separable
            elif camera_inversion_args['type'] == 'non_separable':
                if psf is None:
                    raise NotImplementedError('PSF is None, random init (PSF) not implemented for non-separable dataset')
                
                camera_inversion_args =  dict(camera_inversion_args)
                camera_inversion_args.pop('type')
                camera_inversion_layer = FTLayer(psf=psf, **camera_inversion_args)
            else:
                raise NotImplementedError('Camera inversion type not implemented, choose between separable and non_separable')

        self.camera_inversion_layer = camera_inversion_layer


        if camera_inversion_layer:
            x = camera_inversion_layer(input)
            
        model_config = dict(perceptual_args)
        model_type = model_config.pop('type')

        if model_type == 'unet':
            model_output = [u_net(input=x, **model_config, out_shape=output_shape)]
        else:
            model_output = [experimental_models(model_name=model_config['model_name'], 
                                                input=x, 
                                                out_shape=output_shape,
                                                model_args=model_config['args'])]

        self.perceptual_model = Model(inputs=[x],
                                    outputs=model_output,
                                    name='perceptual_model')
        
        layers = [input]
        if self.camera_inversion_layer:
            if isinstance(self.camera_inversion_layer, tf.keras.Model):
                self.camera_inversion_layer.build(input_shape=self.in_shape)

            layers.append(self.camera_inversion_layer)
        
        layers.append(self.perceptual_model)
        super().__init__(name=name, layers=layers)

        if model_weights_path:
            self.load_weights(model_weights_path).expect_partial()

    
    def summary(self, **kwargs):
        cam_model = None
        if self.camera_inversion_layer:
            inp = Input(shape=self.in_shape, name='input', dtype='float32')
            out = self.camera_inversion_layer(inp)
            cam_model = Model(inputs=[inp], outputs=[out])
            if isinstance(self.camera_inversion_layer, tf.keras.Model):
                self.camera_inversion_layer.summary(**kwargs)
            else:
                cam_model.summary(**kwargs)

        self.perceptual_model.summary(**kwargs)

        if cam_model:
            model = keras.Sequential([cam_model, self.perceptual_model])
            line_length = 98
            print("=" * line_length)
            trainable_count = count_params(model.trainable_weights)
            non_trainable_count = count_params(model.non_trainable_weights)

            print(f"Total params: {trainable_count + non_trainable_count:,}")
            print(f"Trainable params: {trainable_count:,}")
            print(f"Non-trainable params: {non_trainable_count:,}")
            print("_" * line_length)



#################################################################################################################
######################################### Discriminator + GAN ###################################################
#################################################################################################################

######################################### Discriminator #########################################################

class Discriminator(keras.Sequential):
    def __init__(self,
                 input_shape, 
                 filters, 
                 strides, 
                 kernel_size, 
                 activation='swish', 
                 use_groupnorm=False, 
                 num_groups=None, 
                 sigmoid_output=False,
                 name='discriminator'):
        
        assert activation, "activation must be specified"
        input = Input(shape=input_shape, name="input")
        x = input

        assert len(filters) == len(strides) and len(strides) == len(kernel_size)

        for i in range(len(filters)):
            # conv block
            x = Conv2D(filters[i],
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    padding='same',
                    )(x)
            if use_groupnorm:
                x = GroupNormalization(groups=num_groups)(x)
            else:
                x = BatchNormalization()(x)
            
            x = Activation(activation=activation)(x)
            

        x = GlobalAveragePooling2D(keepdims=True)(x)

        x = Conv2D(1, kernel_size=1, padding='same', activation=None)(x)
        x = Reshape(target_shape=[])(x)

        if sigmoid_output:
            x = Activation("sigmoid")(x)
        
        m = Model(inputs=[input], outputs=[x], name='discriminator')
        super().__init__(name=name, layers=m.layers)


######################################### GAN ################################################################

class DiscrLoss(Loss):
    def __init__(self, name='discr_loss', label_smoothing=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, **kwargs)


    def call(self, y_true, y_pred):
        zeros = tf.zeros_like(y_pred)
        ones = tf.ones_like(y_pred)
        if self.label_smoothing:
            zeros = tf.random.uniform(shape=tf.shape(y_pred), 
                                       minval=self.label_smoothing['fake_range'][0], 
                                       maxval=self.label_smoothing['fake_range'][1])
            ones = tf.random.uniform(shape=tf.shape(y_pred), 
                                       minval=self.label_smoothing['true_range'][0], 
                                       maxval=self.label_smoothing['true_range'][1])
            
        real_loss = self.cross_entropy(ones, y_true)
        fake_loss = self.cross_entropy(zeros, y_pred)
        total_loss = real_loss + fake_loss
        return total_loss


class AdversarialLoss(Loss):
    def __init__(self, name='adv_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, **kwargs)

    def call(self, y_true, y_pred):
        loss = self.cross_entropy(tf.ones_like(y_pred), y_pred)

        return loss



class FlatNetGAN(Model):
    def __init__(self, discriminator, generator, global_batch_size=None, label_smoothing=None, **kwargs):
        super(FlatNetGAN, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.global_batch_size = global_batch_size
        self.label_smoothing = label_smoothing

        self.g_adv_loss = AdversarialLoss(name='adv')
        self.d_loss = DiscrLoss(name='discr', label_smoothing=label_smoothing)
        

    def compile(self, optimizer, d_optimizer, lpips_loss, mse_loss, adv_weight, mse_weight, perc_weight, metrics, distributed_gpu=False):
        super(FlatNetGAN, self).compile(metrics=metrics, optimizer=optimizer)
        self.d_optimizer = optimizers.get(d_optimizer) if isinstance(d_optimizer, str) else d_optimizer
        self.g_optimizer = optimizers.get(optimizer) if isinstance(optimizer, str) else optimizer



        self.lpips_loss = lpips_loss
        self.g_mse_loss = mse_loss
        self.adv_weight = adv_weight
        self.mse_weight = mse_weight
        self.perc_weight = perc_weight
        
    def call(self, inputs):
        return self.generator(inputs)


    def train_step(self, inputs):
        sensor_img, real_img = inputs

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_img = self.generator(sensor_img, training=True)
            real_output = self.discriminator(real_img, training=True)
            fake_output = self.discriminator(gen_img, training=True)

            adv_loss = self.g_adv_loss(None, fake_output)
            mse_loss = self.g_mse_loss(real_img, gen_img)
            perc_loss = self.lpips_loss(real_img, gen_img)
            gen_loss = self.adv_weight * adv_loss + self.mse_weight * mse_loss + self.perc_weight * perc_loss

            disc_loss = self.d_loss(real_output, fake_output)
            

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return {"d": disc_loss, "g": gen_loss, "adv": adv_loss, "mse":mse_loss, "lpips" : perc_loss}
    
    def summary(self, **kwargs):

        self.generator.summary(**kwargs)
        self.discriminator.summary(**kwargs)
    
    def get_config(self):
        config = super().get_config()

        config.update({
            "discriminator": self.discriminator, 
            "generator": self.generator, 
            "global_batch_size": self.global_batch_size,
            "label_smoothing": self.label_smoothing
        })
        return config
    






