
import keras
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import AveragePooling2D, Input, concatenate, Lambda
from keras.regularizers import l2
from keras.models import Model

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, learn_bn = True, wd=1e-4, use_relu=True):
    x = inputs
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd), use_bias=False)(x)
    return x


def pad_depth(inputs, desired_channels):
    from keras import backend as K
    y = K.zeros_like(inputs, name='pad_depth1')
    return y


def My_freq_split1(x):
    return x[:,0:64,:,:]


def My_freq_split2(x):
    return x[:,64:128,:,:]


# resnet模型中没有在频域上进行下采样，分成两通道，第一个通道在时域上下采样，第二个不采样
def model_resnet(num_classes, input_shape=[128,None,1], num_filters=24, wd=1e-3):
    
    num_res_blocks=2
    inputs = Input(shape=input_shape)
    Split1 = Lambda(My_freq_split1)(inputs)
    Split2 = Lambda(My_freq_split2)(inputs)

    Residual1 = resnet_layer(inputs=Split1,
                     num_filters=num_filters,
                     strides=[1,2],
                     learn_bn = True,
                     wd=wd,
                     use_relu = False)
    
    Residual2 = resnet_layer(inputs=Split2,
                     num_filters=num_filters,
                     strides=[1,2],
                     learn_bn = True,
                     wd=wd,
                     use_relu = False)

    # Instantiate the stack of residual units
    for stack in range(4):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = [1,2]  # downsample
            Conv1 = resnet_layer(inputs=Residual1,
                             num_filters=num_filters,
                             strides=strides,
                             learn_bn = False,
                             wd=wd,
                             use_relu = True)
            Conv2 = resnet_layer(inputs=Residual2,
                             num_filters=num_filters,
                             strides=strides,
                             learn_bn = False,
                             wd=wd,
                             use_relu = True)
            Conv1 = resnet_layer(inputs=Conv1,
                             num_filters=num_filters,
                             strides=1,
                             learn_bn = False,
                             wd=wd,
                             use_relu = True)
            Conv2 = resnet_layer(inputs=Conv2,
                             num_filters=num_filters,
                             strides=1,
                             learn_bn = False,
                             wd=wd,
                             use_relu = True)
            if stack > 0 and res_block == 0:  
                # average pool and downsample the residual path
                Residual1 = AveragePooling2D(pool_size=(3, 3), strides=[1,2], padding='same')(Residual1)
                Residual2 = AveragePooling2D(pool_size=(3, 3), strides=[1,2], padding='same')(Residual2)
                
                # zero pad to increase channels
                desired_channels = Conv1.shape.as_list()[-1]

                Padding1=Lambda(pad_depth,arguments={'desired_channels':desired_channels})(Residual1)
                Residual1 = keras.layers.Concatenate(axis=-1)([Residual1,Padding1])
                
                Padding2=Lambda(pad_depth,arguments={'desired_channels':desired_channels})(Residual2)
                Residual2 = keras.layers.Concatenate(axis=-1)([Residual2,Padding2])

            Residual1 = keras.layers.add([Conv1,Residual1])
            Residual2 = keras.layers.add([Conv2,Residual2])
            
        num_filters *= 2
        

    Residual = concatenate([Residual1,Residual2],axis=1)
    Output = resnet_layer(inputs=Residual,
                             num_filters=2*num_filters,
                              kernel_size=1,
                             strides=1,
                             learn_bn = False,
                             wd=wd,
                             use_relu = True)
    #output layers after last sum
    Output = resnet_layer(inputs=Output,
                     num_filters=num_classes,
                     strides = 1,
                     kernel_size=1,
                     learn_bn = False,
                     wd=wd,
                     use_relu=False)
    Output = BatchNormalization(center=False, scale=False)(Output)
    Output = GlobalAveragePooling2D()(Output)
    Output = Activation('softmax')(Output)

    model = Model(inputs=inputs, outputs=Output)
    return model
