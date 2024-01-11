import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(t_images, t_labels), (test_images,
                       test_labels) = fashion_mnist.load_data()  # split automatically
name_class = ['Trouser', 'Shirt', 'T-shirt/top',  'Dress', 'Coat', 'Ankle boot',
              'Sandal', 'Sneaker', 'Bag', 'Pullover',]
t_images = t_images/255
test_images = test_images/255

model = keras.Sequential(
    [keras.layers.Flatten(input_shape=(28, 28)),  # input ---->> flatten all the pixel in 28*28
     # hidden 1st-->> why 128 --->>
     keras.layers.Dense(128, activation='relu'),
     keras.layers.Dense(25, activation='relu'),
     keras.layers.Dense(10, activation='softmax')
     ]  # output layer -->> softmax (all sums up to 1)
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # adam is gradient descent algo used to do that
              # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
              loss='SparseCategoricalCrossentropy',
              metrics=['accuracy'])


model.fit(t_images, t_labels, epochs=10)
tested_loss, test_accuracy = model.evaluate(
    test_images, test_labels, verbose=1)

# improovement

model = keras.Sequential(
    [keras.layers.Flatten(input_shape=(28, 28)),  # input ---->> flatten all the pixel in 28*28
     # hidden 1st-->> why 128 --->>
     keras.layers.Dense(128, activation='relu'),
     keras.layers.Dense(25, activation='relu'),
     keras.layers.Dense(10, activation='linear')
     ]  # output layer -->> softmax (all sums up to 1)
)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # adam is gradient descent algo used to do that
              #
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

logits = model(t_images)
predictions = tf.nn.softmax(logits)
model.fit(t_images, t_labels, epochs=10)
tested_loss, test_accuracy = model.evaluate(
    test_images, test_labels, verbose=1)


# running whole model :-

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR


def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Excpected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    print("\n ")
    print("****************************************************************")
    print("<-------------IMAGE YOU SELECTED TO PREDICT------------------->")
    print("*******************************************************************")
    show_image(image, class_names[correct_label], predicted_class)
    print("*******************************************************************")
    print("THE ACTUAL CLASS OF THE IMAGE :-", class_names[correct_label])
    print("*******************************************************************")
    print("THE PREDICTED CLASS OF THE IMAGE:-", predicted_class)
    print("*******************************************************************")


def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
        else:
            print("Try again...")


num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
