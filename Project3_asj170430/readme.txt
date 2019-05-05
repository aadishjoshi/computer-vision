/***************************************************************************************************
*Written By:- Aadish Joshi
*MNIST digit recognizer
***************************************************************************************************/

Model documentation:

Layers used:-
	tf.keras.layers.MaxPool2D(4, 4, input_shape=(28,28,1)),
	tf.keras.layers.Conv2D(7, (3,3), padding='same', activation=tf.nn.relu),
	tf.keras.layers.MaxPool2D(2, 2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation=tf.nn.relu),
	tf.keras.layers.Dense(10, activation=tf.nn.softmax)

Accuracy hit: 93.10%