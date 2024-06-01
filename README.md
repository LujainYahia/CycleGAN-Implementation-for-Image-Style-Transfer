# CycleGAN Implementation for Image Style Transfer

## Overview
This repository contains an implementation of a CycleGAN model using TensorFlow and Keras. The CycleGAN model is designed for image style transfer tasks, where the goal is to transform images from one domain to another and vice versa, without paired examples. This implementation includes custom layers, generator and discriminator models, training functions, and visualization tools.

## Contents
- `InstanceNormalization` Layer: A custom layer for instance normalization.
- `downsample` and `upsample` Functions: Building blocks for the U-Net generator.
- `unet_generator` Function: Constructs the U-Net generator model.
- `discriminator` Function: Constructs the discriminator model.
- Training and Loss Functions: Functions to train the CycleGAN model and compute losses.
- Visualization Tools: Functions to visualize the results and training progress.
- Checkpointing and GIF Creation: Mechanisms to save model checkpoints and create GIFs of the training progress.

## Requirements
- TensorFlow 2.x
- Matplotlib
- Imageio
- Glob

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cyclegan-tensorflow.git
   cd cyclegan-tensorflow
   ```
2. Install the required packages:
   ```bash
   pip install tensorflow matplotlib imageio
   ```

## Usage
### Training the CycleGAN Model
1. **Prepare Data**: Ensure you have two datasets representing two different styles. These should be in the form of TensorFlow datasets.
2. **Define Model Components**:
   ```python
   generator_g = unet_generator()
   generator_f = unet_generator()
   discriminator_x = discriminator()
   discriminator_y = discriminator()
   ```
3. **Sample Data**:
   ```python
   sample_style1_data = tf.random.normal([1, 128, 128, 1])
   sample_style2_data = tf.random.normal([1, 128, 128, 1])
   ```
4. **Train the Model**:
   ```python
   for epoch in range(1, EPOCHS+1):
       for image_x, image_y in tf.data.Dataset.zip((style1_data, style2_data)):
           train_step(image_x, image_y)
       generate_images(generator_g, sample_style1_data, generator_f, sample_style2_data)
       ckpt_save_path = ckpt_manager.save()
       print('Saving checkpoint for epoch', epoch, 'at', ckpt_save_path)
   ```
5. **Generate GIF**:
   ```python
   anim_file = 'cyclegan.gif'
   with imageio.get_writer(anim_file, mode='I') as writer:
       filenames = glob.glob('image*.png')
       filenames = sorted(filenames)
       for filename in filenames:
           image = imageio.imread(filename)
           writer.append_data(image)
       image = imageio.imread(filename)
       writer.append_data(image)
   ```

### Visualization
Visualize the results after training using:
```python
def generate_images(model1, test_input1, model2, test_input2):
    prediction1 = model1(test_input1)
    prediction2 = model2(test_input2)
    plt.figure(figsize=(8, 4))
    display_list = [test_input1[0], prediction1[0], test_input2[0], prediction2[0]]
    title = ['Input Image', 'Predicted Image', 'Input Image', 'Predicted Image']
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i].numpy()[:, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
```

## Detailed Code Explanation
### InstanceNormalization Layer
This custom layer normalizes the input images on a per-instance basis, making it suitable for style transfer tasks.
```python
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
```

### Downsample and Upsample Functions
These functions create the encoder and decoder parts of the U-Net generator.
```python
def downsample(filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    result.add(InstanceNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result
```

### U-Net Generator
The generator model is built using downsample and upsample blocks.
```python
def unet_generator():
    down_stack = [
        downsample(64, 4, False), 
        downsample(128, 4), 
        downsample(128, 4), 
        downsample(128, 4), 
        downsample(128, 4)
    ]
    up_stack = [
        upsample(128, 4, True), 
        upsample(128, 4, True), 
        upsample(128, 4), 
        upsample(64, 4)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', kernel_initializer=initializer,
                                           activation='tanh')
    concat = tf.keras.layers.Concatenate()
    inputs = tf.keras.layers.Input(shape=[128, 128, 1])
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
```

### Discriminator
The discriminator model classifies images as real or fake.
```python
def discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[128, 128, 1], name='input_image')
    x = inp
    down1 = downsample(64, 4, False)(x) 
    down2 = downsample(128, 4)(down1) 
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2) 
    conv = tf.keras.layers.Conv2D(256, 4, strides=1, kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)
    norm1 = InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=inp, outputs=last)
```

### Loss Functions and Training Step
Define loss functions and the training step.
```python
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return 10.0 * loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)

@tf.function
def
