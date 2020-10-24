import keras
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, MaxPooling2D, Dense
from keras.layers import Input, Dropout, ZeroPadding2D
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K
import sys
sys.path.append("..")
import attention_layer


def output_layer(inputs, num_filters=16, kernel_size=3, strides=1, learn_bn = True, wd=1e-4, use_relu=True):
    x = inputs
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='valid', kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center = learn_bn, scale = learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    return x

# (48, 96, 512, 768)
# (144, 288, 1536, 2304)
def conv_layer_1(inputs, num_channels=6, num_filters=14, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size1 = [5, 5]
    kernel_size2 = [3, 3]
    strides1 = [1, 2]
    strides2 = [1, 1]
    num_channels = 1

    x = inputs
    x = BatchNormalization(center = learn_bn, scale = learn_bn)(x)

    # 1 
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size1, strides=strides1,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center = learn_bn, scale = learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    # 2
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size2, strides=strides2,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3),strides=[2,2], padding='same')(x)
    return x


def conv_layer_2(inputs, num_channels=6, num_filters=28, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    num_channels = 1

    x = inputs

    #1
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    #2
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)


    x = MaxPooling2D(pool_size=(3, 3), strides=[2,2], padding='same')(x)
    return x

def conv_layer_3(inputs, num_channels=6, num_filters=56, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    num_channels = 1


    x = inputs
    # 1
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    # x = Dropout(0.2)(x)
    #2
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
 
    # Max Pooling
    x = MaxPooling2D(pool_size=(3, 3), strides=[1,2], padding='same')(x) 

    # 3
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    # x = Dropout(0.2)(x)
    #4
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=[1, 2], padding='same')(x)
    return x


def conv_layer_4(inputs, num_channels=6, num_filters=128, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    num_channels = 1

    x = inputs

    # 1
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    # 2
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    return x


def model_fsfcnn(num_classes, input_shape=[128, None, 3], num_filters=[14, 24, 48, 96], wd=1e-3):

    inputs = Input(shape = input_shape)

    
    Conv1 = conv_layer_1(inputs = inputs,
                            num_channels = input_shape[-1],
                            num_filters = num_filters[0],
                            learn_bn = True,
                            wd = wd,
                            use_relu = True)

    Conv2 = conv_layer_2(inputs = Conv1,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[1],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    Conv3 = conv_layer_3(inputs=Conv2,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[2],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)

    Conv4 = conv_layer_4(inputs=Conv3,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[3],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)

    # output layers after last sum
    Output = output_layer(inputs = Conv4,
                              num_filters = num_classes,
                              strides = 1,
                              kernel_size = 1,
                              learn_bn = False,
                              wd = wd,
                              use_relu = True)

    Output = BatchNormalization(center = False, scale = False)(Output)
    Output = attention_layer.channel_attention(Output, ratio = 2)

    # 使用全局平均池化操作，代替了全连接层，因此使用channel_attention layer关注channel维度的特征
    Output = GlobalAveragePooling2D()(Output)
    Output = Activation('softmax')(Output)

    # Instantiate model.
    model = Model(inputs = inputs, outputs = Output)
    return model