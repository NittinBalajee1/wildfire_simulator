import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from . import config

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Dice coefficient metric for model training."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function for model training."""
    return 1.0 - dice_coefficient(y_true, y_pred)

def conv_block(input_tensor, num_filters):
    """Create a convolutional block with two conv layers."""
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def create_unet_model(input_shape, num_classes):
    """Create a U-Net model for wildfire spread prediction."""
    inputs = Input(input_shape)
    
    # Contracting Path (Encoder)
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottom
    c5 = conv_block(p4, 1024)
    
    # Expansive Path (Decoder)
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 512)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 256)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 128)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 64)
    
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def get_model():
    """Create and compile the U-Net model with appropriate loss and metrics."""
    model = create_unet_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES
    )
    
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE),
                 loss=dice_loss,
                 metrics=[dice_coefficient, 'accuracy', tf.keras.metrics.MeanIoU(num_classes=config.NUM_CLASSES)])
    
    return model

def get_callbacks():
    """Create and return training callbacks."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(os.path.join(config.MODEL_DIR, 'best_model.h5'), 
                       save_best_only=True, 
                       monitor='val_dice_coefficient',
                       mode='max',
                       verbose=1),
        EarlyStopping(monitor='val_dice_coefficient',
                     patience=config.EARLY_STOPPING_PATIENCE,
                     mode='max',
                     restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss',
                         factor=0.2,
                         patience=config.REDUCE_LR_PATIENCE,
                         min_lr=1e-6)
    ]
    
    return callbacks
