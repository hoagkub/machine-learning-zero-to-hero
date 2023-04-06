- [MACHINE LEARNING ZERO TO HERO](#machine-learning-zero-to-hero)
  - [BASIC STEPS](#basic-steps)
    - [1. IMPORT TENSORFLOW LIBRARY](#1-import-tensorflow-library)
    - [2. DEFINE LAYERS OF MODEL](#2-define-layers-of-model)
      - [2.1 THE SEQUENTIAL CLASS](#21-the-sequential-class)
  - [REFERENCES](#references)
    - [1. FOR "DEFINE LAYERS OF MODEL" SECTION](#1-for-define-layers-of-model-section)
      - [Arguments](#arguments)

# MACHINE LEARNING ZERO TO HERO

## BASIC STEPS

### 1. IMPORT TENSORFLOW LIBRARY

**`Lab2-Basic-Computer-Vision.ipynb`**

```python
import tensorflow as tf
print(tf.__version__)
```

### 2. DEFINE LAYERS OF MODEL

#### [2.1 THE SEQUENTIAL CLASS](https://keras.io/api/models/sequential/)

**Best practice examples:**

**`Lab1-Intro-to-ML.ipynb`**

```python
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
```

**`Lab2-Basic-Computer-Vision.ipynb`**

```python
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
```

**`Lab3-Introducing-Convolutional-Neural-Network.ipynb`**

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

**`Lab4-Build-an-Image-Classifer.ipynb`**

```python
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

## REFERENCES

### 1. FOR "DEFINE LAYERS OF MODEL" SECTION
* [Keras layers API](https://keras.io/api/layers/). Follows are frequently used layers.
  + [Core layers (includes: Dense, etc.)](https://keras.io/api/layers/core_layers/)
  + [Reshaping layers (includes: Flatten, etc.)](https://keras.io/api/layers/reshaping_layers/)
  + [Pooling layers](https://keras.io/api/layers/pooling_layers/)
    - Pooling layers are used to reduce the dimensions of the feature maps.
  + [Regularization Layers](https://keras.io/api/layers/regularization_layers/)
    - The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.

#### Arguments

1. Optimizer: (trans: thuật toán tối ưu hóa)
  - sgd: stochastic gradient descent
- Loss: hàm mất mát, bao gồm
  - mean_squared_error

___Obtain from: "https://www.youtube.com/watch?v=NVsw-JrXv9I&list=PLQY2H8rRoyvxNqk9EV5VP5fS0cWEXW5QQ"___
