- [MACHINE LEARNING ZERO TO HERO](#machine-learning-zero-to-hero)
  - [BASIC STEPS](#basic-steps)
    - [1. IMPORT TENSORFLOW LIBRARY](#1-import-tensorflow-library)
    - [2. DEFINE LAYERS OF MODEL. USING THE SEQUENTIAL CLASS](#2-define-layers-of-model-using-the-sequential-class)
    - [3. CONFIGURE THE MODEL FOR TRAINING. USING Model.compile(...)](#3-configure-the-model-for-training-using-modelcompile)
    - [4. TRAIN MODEL. USING Model.fit(...)](#4-train-model-using-modelfit)
    - [5. EVALUATE MODEL. USING Medel.evaluate(...)](#5-evaluate-model-using-medelevaluate)
  - [REFERENCES](#references)
    - [1. FOR "DEFINE LAYERS OF MODEL" SECTION](#1-for-define-layers-of-model-section)
    - [2. FOR "CONFIGURE MODEL" AND "TRAN MODEL" SECTION](#2-for-configure-model-and-tran-model-section)
    - [3. ML zero to hero](#3-ml-zero-to-hero)

# MACHINE LEARNING ZERO TO HERO

## BASIC STEPS

### 1. IMPORT TENSORFLOW LIBRARY

**`Lab2-Basic-Computer-Vision.ipynb`**

```python
import tensorflow as tf
print(tf.__version__)
```

### 2. DEFINE LAYERS OF MODEL. USING [THE SEQUENTIAL CLASS](https://keras.io/api/models/sequential/)

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

### 3. CONFIGURE THE MODEL FOR TRAINING. USING [Model.compile(...)](https://keras.io/api/models/model_training_apis/#compile-method)

**Popular *args**

* **optimizer** (trans: thuật toán tối ưu hóa)
  + See [tf.keras.optimizers](https://keras.io/api/optimizers/)
* **loss** (trans: hàm mất mát)
  + See [tf.keras.losses](https://keras.io/api/losses/)
* **metrics**
  + See [tf.keras.metrics](https://keras.io/api/metrics/)

### 4. TRAIN MODEL. USING [Model.fit(...)](https://keras.io/api/models/model_training_apis/#fit-method)

**Popular *args**

* **x**
* **y**
* **epochs**

### 5. EVALUATE MODEL. USING [Medel.evaluate(...)](https://keras.io/api/models/model_training_apis/#evaluate-method)

**Popular *args**

* **x**
* **y**

## REFERENCES

### 1. FOR "DEFINE LAYERS OF MODEL" SECTION
* [Keras layers API](https://keras.io/api/layers/). Follows are frequently used layers.
  + [Core layers (includes: Dense, etc.)](https://keras.io/api/layers/core_layers/)
  + [Reshaping layers (includes: Flatten, etc.)](https://keras.io/api/layers/reshaping_layers/)
  + [Pooling layers](https://keras.io/api/layers/pooling_layers/)
    - Pooling layers are used to reduce the dimensions of the feature maps.
  + [Regularization Layers](https://keras.io/api/layers/regularization_layers/)
    - The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
### 2. FOR "CONFIGURE MODEL" AND "TRAN MODEL" SECTION
* [Model training APIs](https://keras.io/api/models/model_training_apis)

### 3. [ML zero to hero](https://www.youtube.com/watch?v=NVsw-JrXv9I&list=PLQY2H8rRoyvxNqk9EV5VP5fS0cWEXW5QQ)
