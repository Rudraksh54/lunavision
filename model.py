import tensorflow as tf
import numpy as np 

def circular_mask(kernel_size):
    #helper fucntion
    y, x = np.ogrid[:kernel_size, :kernel_size]
    center = (kernel_size - 1) / 2
    dist_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    radius = kernel_size / 2
    mask = dist_from_center <= radius
    return mask.astype(np.float32)

#custum layer for detecting circular geometry 
class CircularConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', **kwargs):
        super(CircularConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        # Define convolutional weights (will be masked)
        self.kernel = self.add_weight(
            shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
            initializer="glorot_uniform",
            trainable=True,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.filters,),
            initializer="zeros",
            trainable=True,
            name='bias'
        )

        # Creation of  circular mask (broadcasted over channels)
        mask = circular_mask(self.kernel_size)  # shape (k, k)
        mask = np.expand_dims(mask, axis=-1)  # shape (k, k, 1)
        mask = np.expand_dims(mask, axis=-1)  # shape (k, k, 1, 1)
        self.mask = tf.constant(mask, dtype=tf.float32)  # shape (k, k, 1, 1)

    def call(self, inputs):
        masked_kernel = self.kernel * self.mask  # Applying circular mask
        return tf.nn.conv2d(
            inputs,
            masked_kernel,
            strides=[1, *self.strides, 1],
            padding=self.padding.upper()
        ) + self.bias
        
def resnet_circular_identity_block(X, f, filters):
    # skips over 2 (one of them being cirluarcon2d) layers and adds the first to last
    (F1,F2,F3) = filters
    X_short = X
    # first layer
    X = tf.keras.layers.Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid')(X)
    X =tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layersActivation('relu')(X)
    #second layer
    X= CircularConv2D(filters=F2, kernel_size=f)(X)
    X =tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layersActivation.mish(X)
    #third 
    X = tf.keras.layers.Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid')(X)
    X =tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Add()([X_short , X])
    X = tf.kears.layers.Activation('relu')(X)
    return X
def resnet_convblock(X, f, filters, s):
    (F1,F2,F3) = filters
    X_shortcut = X
    # first layer
    X = tf.keras.layers.Conv2D(filters = F1, kernel_size = 1, strides = (s,s), padding = 'valid')(X)
    X =tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layersActivation('relu')(X)
    # second 
    X =tf.keras.layers.Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation.leaky_relu(alpha=0.2)(X)
    #third
    X =tf.keras.layers.Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding='valid')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    #shortcut path 
    X_shortcut =tf.keras.layers.Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding='valid')(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis = 3)(X_shortcut)
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    return X
    
def lunavis_model(input_shape=(416,416,3)):
    inputs = tf.keras.Input(shape=input_shape)
    #basic cnn 
    for i in range(5):
     X = resnet_circular_identity_block(inputs,6,(16,32,32))
    for i in range (3):
     X = resnet_convblock(X, 3, (32,64,64),2)
    for i in range(8):
     X =tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides = (1,1), padding='same')(X)
     X = tf.keras.layers.Activation.leaky_relu( alpha = 0.1)(X)
     X = tf.keras.layers.BatchNormalization(axis = 3)(X)
     X =tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides = (1,1), padding='same')(X)
     X = tf.keras.layers.Activation.mish(X)
     X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = resnet_convblock(X, 3, (128,256,256),1)
    for i in range(5):
     X = resnet_circular_identity_block(X,3,(256,512,512))
    X = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(X)
    for i in range(3):
     X = resnet_convblock(X, 3, (512,1024,1024),1)
    X = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(X)
    X = resnet_convblock(X, 3, (1024,1024,1024),2)
    #final dense layer in conv form 
    X = tf.keras.layers.Conv2D(5, (1, 1), padding='same', activation='linear')(X)
    outputs = tf.keras.layers.Reshape((26,26,5))(X)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


#custom loss fucntion for yolo like algo
def crater_detection_loss(y_true, y_pred):
    """
    y_true, y_pred: shape (batch, 26, 26, 5)
    """
    obj_mask = y_true[..., 0:1]  # Shape: (B, 26, 26, 1)

    # Objectness loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    obj_loss = bce(y_true[..., 0], y_pred[..., 0])

    # Coordinate losses (only where object exists)
    pred_xy = tf.sigmoid(y_pred[..., 1:3])  # offset prediction
    true_xy = y_true[..., 1:3]

    xy_loss = tf.reduce_sum(obj_mask * tf.square(pred_xy - true_xy))

    pred_wh = y_pred[..., 3:5]  # predicted log(w), log(h)
    true_wh = y_true[..., 3:5]
    wh_loss = tf.reduce_sum(obj_mask * tf.square(pred_wh - true_wh))

    total_loss = obj_loss + 5 * xy_loss + 1 * wh_loss
    return total_loss
model = lunavis_model()
# First 30 epochs: warm-up with Adam
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=crater_detection_loss)
model.fit()

# Then fine-tune last 20 epochs with SGD
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9), loss=crater_detection_loss)
model.fit()
