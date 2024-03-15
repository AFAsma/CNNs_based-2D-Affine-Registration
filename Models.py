import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import VGG16
import keras.regularizers as regularizers
import TransformationSpatial as Trans
from keras.layers import Lambda

class AffineConvModel:
    """
    Tang, K., Li, Z., Tian, L., Wang, L., Zhu, Y., 2020. ADMIR–Affine and Deformable Medical Image Registration for
    Drug-Addicted Brain Images. IEEE Access 8, 70960–70968. doi:10.1109/ACCESS.2020.2986829.
        Parameters:
            inshape: Input shape. e.g. (256,256)
    """
    def __init__(self, input_shape=(256, 256),batch_norm_mode=None):
        self.model = self._build_model(input_shape,batch_norm_mode)     
          
    def _build_model(self, input_shape, batch_norm_mode=None):
        static_input = tf.keras.Input(shape=input_shape + (1,), name='static')
        moving_input = tf.keras.Input(shape=input_shape + (1,), name='moving')
        x = layers.concatenate([static_input, moving_input], axis=-1)

        # Define batch normalization 
        batch_norm_layer = (layers.BatchNormalization(axis=-1) if batch_norm_mode == "before_act"
                            else lambda x: x)
        
        # Strided convolution layer, use to reduce the number of channel => reduce weight number
        encoder_layers = [
            layers.Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu'),
            batch_norm_layer,
            layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu'),
            batch_norm_layer,
            layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'),
            batch_norm_layer,  
            layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
            batch_norm_layer,
            layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'),
            batch_norm_layer,
            layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu'),
            layers.Flatten()
       ]
        # Classification layer to get the transformation matrix 2*3
        classification_layer = layers.Dense(6, kernel_initializer='zeros',
                                             bias_initializer=tf.constant_initializer([1, 0, 0, 0, 1, 0]),
                                             kernel_regularizer='l2')

        for layer in encoder_layers:
            x = layer(x)
        x = classification_layer(x)

        # build transformer layer
        nb, _ = tf.shape(x)
        theta_layer = layers.Reshape((2, 3))
        theta = theta_layer(x)
        grid = Trans.regular_grid_2d(256, 256)
        grid_new = Trans.grid_transform(theta, grid)
        grid_new = tf.clip_by_value(grid_new, -1, 1)

        # warp the moving image with the transformer
        moved = Trans.grid_sample_2d(moving_input, grid_new)

        model = tf.keras.Model(inputs=[static_input, moving_input], outputs=moved, name='max_cnn_model')
        return model

    def register_images(self, static_image, moving_image):
        """
        Predicts the transform of moving tensors.
        """
        return self.model([static_image, moving_image])


def conv_block(x, nb_filter):

    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(nb_filter, 3, 1, use_bias=False, padding="SAME")(x)

    return x
def dense_block(x, nb_layers, nb_filter, growth_rate, grow_nb_filters=True):
    concat_feat = x
    for i in range(nb_layers):
        x = conv_block(concat_feat, growth_rate)
        concat_feat = tf.concat([concat_feat, x], -1)

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
def DenseBlocks(x,nb_filter=8, nb_layers = 1,growth_rate=8):
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate)
    return x

class AirCNNModel:
    """
    Chee, E., Wu, Z., 2018. AIRNet: Self-Supervised Affine Registration for 3D Medical Images using Neural Networks.
    CoRR abs/1810.02583. URL: http://arxiv.org/abs/1810.02583, arXiv:1810.02583.
    """
    def __init__(self, input_shape=(256, 256), in_channels=1,batch_norm_mode=None):
         self.model = self._build_model(input_shape,in_channels,batch_norm_mode)
        
    def _build_model(self,input_shape,in_channels=1,batch_norm_mode=None):
        num_filters = [8, 16, 32, 64, 128, 256]

        input_shape = input_shape + (in_channels,)
        moving = tf.keras.layers.Input(shape=input_shape, name='moving')
        static = tf.keras.layers.Input(shape=input_shape, name='static')
  
       # Define batch normalization 
        batch_norm_layer = (layers.BatchNormalization(axis=-1) if batch_norm_mode == "before_act"
                            else lambda x: x)

        out_1_1 = layers.Conv2D(num_filters[0], kernel_size=3, padding='same', activation='relu')(static)
        out_1_1=batch_norm_layer(out_1_1)
        out_1_1 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_1_1)
        out_1_1 = DenseBlocks(out_1_1, nb_filter=8, nb_layers=1)


        out_2_1 = layers.Conv2D(num_filters[0], kernel_size=3, padding='same', activation='relu')(moving)
        out_2_1=batch_norm_layer(out_2_1)
        out_2_1 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_2_1)
        out_2_1 = DenseBlocks(out_2_1, nb_filter=8, nb_layers=1)


        out_1_2 = layers.Conv2D(num_filters[1], kernel_size=1, padding='same', activation='relu')(out_1_1)
        out_1_2=batch_norm_layer(out_1_2)
        out_1_2 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_1_2)
        out_1_2 = DenseBlocks(out_1_2, nb_filter=8, nb_layers=2)


        out_2_2 = layers.Conv2D(num_filters[1], kernel_size=1, padding='same', activation='relu')(out_2_1)
        out_2_2=batch_norm_layer(out_2_2)
        out_2_2 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_2_2)
        out_2_2 = DenseBlocks(out_2_2, nb_filter=8, nb_layers=2)



        out_1_3 = layers.Conv2D(num_filters[2], kernel_size=1, padding='same', activation='relu')(out_1_2)
        out_1_3=batch_norm_layer(out_1_3)
        out_1_3 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_1_3)
        out_1_3 = DenseBlocks(out_1_3, nb_filter=8, nb_layers=4)

        out_2_3 = layers.Conv2D(num_filters[2], kernel_size=1, padding='same', activation='relu')(out_2_2)
        out_2_3=batch_norm_layer(out_2_3)
        out_2_3 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_2_3)
        out_2_3 = DenseBlocks(out_2_3, nb_filter=8, nb_layers=4)


        out_1_4 = layers.Conv2D(num_filters[3], kernel_size=1, padding='same', activation='relu')(out_1_3)
        out_1_4=batch_norm_layer(out_1_4)
        out_1_4 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_1_4)
        out_1_4 = DenseBlocks(out_1_4, nb_filter=8, nb_layers=8)

        out_2_4 = layers.Conv2D(num_filters[3], kernel_size=1, padding='same', activation='relu')(out_2_3)
        out_2_4=batch_norm_layer(out_2_4)
        out_2_4 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_2_4)
        out_2_4 = DenseBlocks(out_2_4, nb_filter=8, nb_layers=8)


        out_1_5 = layers.Conv2D(num_filters[4], kernel_size=1, padding='same', activation='relu')(out_1_4)
        out_1_5=batch_norm_layer(out_1_5)
        out_1_5 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_1_5)
        out_1_5 = DenseBlocks(out_1_5, nb_filter=8, nb_layers=16)

        out_2_5 = layers.Conv2D(num_filters[4], kernel_size=1, padding='same', activation='relu')(out_2_4)
        out_2_5=batch_norm_layer(out_2_5)
        out_2_5 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_2_5)
        out_2_5 = DenseBlocks(out_2_5, nb_filter=8, nb_layers=16)


        out_1_6 = layers.Conv2D(num_filters[5], kernel_size=1, padding='same', activation='relu')(out_1_5)
        out_1_6=batch_norm_layer(out_1_6)
        out_1_6 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_1_6)
        out_1_6 = DenseBlocks(out_1_6, nb_filter=8, nb_layers=32)

        out_2_6 = layers.Conv2D(num_filters[5], kernel_size=1, padding='same', activation='relu')(out_2_5)
        out_2_6=batch_norm_layer(out_2_6)
        out_2_6 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_2_6)
        out_2_6 = DenseBlocks(out_2_6, nb_filter=8, nb_layers=32)


        out_1_7 = layers.Conv2D(num_filters[5], kernel_size=1, padding='same', activation='relu')(out_1_6)
        out_1_7=batch_norm_layer(out_1_7)
        out_1_7 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_1_7)

        out_2_7 = layers.Conv2D(num_filters[5], kernel_size=1, padding='same', activation='relu')(out_2_6)
        out_2_7=batch_norm_layer(out_2_7)
        out_2_7 = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(out_2_7)

        x_static = layers.Flatten()(out_1_7)
        y_moving = layers.Flatten()(out_2_7)

        x_concat = layers.concatenate([x_static, y_moving], axis=-1)

        fc_1 = layers.Dense(1024, activation='relu')(x_concat) 
        fc_1 = batch_norm_layer(fc_1)
        fc_2 = layers.Dense(512, activation='relu')(fc_1) 
        fc_2 = batch_norm_layer(fc_2)
        fc_3 = layers.Dense(256, activation='relu')(fc_2) 
        fc_3 = batch_norm_layer(fc_3)
        fc_4 = layers.Dense(128, activation='relu')(fc_3) 
        fc_4 = batch_norm_layer(fc_4)
        fc_5 = layers.Dense(64, activation='relu')(fc_4) 
        fc_5 = batch_norm_layer(fc_5)
        x_concat = layers.Dense(6, kernel_initializer='zeros',
                                bias_initializer=tf.constant_initializer([1, 0, 0, 0, 1, 0]), kernel_regularizer='l2')(fc_5)

        nb, _ = tf.shape(x_concat)
        theta = tf.reshape(x_concat, [nb, 2, 3])
        grid = Trans.regular_grid_2d(256, 256)
        grid_new = Trans.grid_transform(theta, grid)
        grid_new = tf.clip_by_value(grid_new, -1, 1)
        moved = Trans.grid_sample_2d(moving, grid_new)

        model = tf.keras.Model(inputs=[static, moving], outputs=moved, name='air_cnn_model')
        return model
class VGG_CNN16(tf.keras.Model):
    def __init__(self, input_shape=(256,256)):
        super(VGG_CNN16, self).__init__()

        in_channels = 1
        self.inputs_shape = input_shape + (in_channels,)
        moving = layers.Input(shape=self.inputs_shape, name='moving')
        static = layers.Input(shape=self.inputs_shape, name='static')

        # Concatenate static and moving images along channel axis
        xx = layers.concatenate([static, static], axis=-1)
        x_in = layers.concatenate([xx, moving], axis=-1)

        # Load pre-trained VGG16 model
        vgg16 = VGG16(weights=None, include_top=False, input_shape=(256,256,3))

        # Freeze VGG16 layers
        for layer in vgg16.layers:
            layer.trainable = False

        # Pass the concatenated input through VGG16 layers
        x = vgg16(x_in)

        # Flatten the output
        x = layers.Flatten()(x)

        # Add dense layers for transformation parameters prediction
        x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dense(6, kernel_initializer='zeros', bias_initializer=tf.constant_initializer([1,0,0,0,1,0]), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

        # Reshape transformation parameters
        nb, _ = tf.shape(x)
        theta = tf.reshape(x, [nb, 2, 3])
        grid = Trans.regular_grid_2d(256, 256)
        grid_new = Trans.grid_transform(theta, grid)
        grid_new = tf.clip_by_value(grid_new, -1, 1)
        moved = Trans.grid_sample_2d(moving, grid_new)

        # Define model
        self.model = tf.keras.Model(inputs=[static, moving], outputs=moved, name='vgg_cnn16')

    def call(self, inputs):
        return self.model(inputs)
class denseNet121_cnn_(tf.keras.Model):
    def __init__(self, input_shape=(256,256)):
        super(VGG_CNN16, self).__init__()

        in_channels = 1
        self.inputs_shape = input_shape + (in_channels,)
        moving = layers.Input(shape=self.inputs_shape, name='moving')
        static = layers.Input(shape=self.inputs_shape, name='static')

        # Concatenate static and moving images along channel axis
        xx = layers.concatenate([static, static], axis=-1)
        x_in = layers.concatenate([xx, moving], axis=-1)

        # Load pre-trained VGG16 model
        densenet = DenseNet121(weights=None, include_top=False, input_shape=(256,256,3))

        # Freeze VGG16 layers
        for layer in densenet.layers:
            layer.trainable = False

        # Pass the concatenated input through VGG16 layers
        x = densenet(x_in)

        # Flatten the output
        x = layers.Flatten()(x)

        # Add dense layers for transformation parameters prediction
        x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dense(6, kernel_initializer='zeros', bias_initializer=tf.constant_initializer([1,0,0,0,1,0]), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

        # Reshape transformation parameters
        nb, _ = tf.shape(x)
        theta = tf.reshape(x, [nb, 2, 3])
        grid = Trans.regular_grid_2d(256, 256)
        grid_new = Trans.grid_transform(theta, grid)
        grid_new = tf.clip_by_value(grid_new, -1, 1)
        moved = Trans.grid_sample_2d(moving, grid_new)

        # Define model
        self.model = tf.keras.Model(inputs=[static, moving], outputs=moved, name='denseNet121_cnn')

    def call(self, inputs):
        return self.model(inputs)
def main():
    registration_model = Model() #e.g: Affine ConvNet
    # Create inputs
    static_image = tf.random.normal((1, 256, 256, 1))
    moving_image = tf.random.normal((1, 256, 256, 1))
    # Call the model 
    registered_image = registration_model.register_images(static_image, moving_image)

    print("Registration complete.")
    #registration_model.model.summary()
if __name__ == "__main__":
    main()
