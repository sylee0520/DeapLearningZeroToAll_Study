import tensorflow as tf

learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalizing data
x_train, x_test = x_train / 255.0, x_test / 255.0

# change data shape
print(x_train.shape) # (60000, 28, 28)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

# change result to one-hot encoding
# in tf1, one_hot=True in read_data_sets("MNIST_data/", one_hot=True)
# took care of it, but here we need to manually convert them
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=10, input_dim=784, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(0.001), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

predictions = tf.model.predict(x_test)
print('Prediction: \n', predictions)
score = tf.model.evaluate(x_test, y_test)
print('Accuracy: ', score[1])

'''
- epoch 와 batch 의 차이
epoch 는 전체 데이터를 한번 학습 시키는 것
batch 는 전체 데이터 양이 너무 많으므로 쪼개서 학습시키는데, 그 쪼개지는 단위

ex) 1000개의 training data 가 있을 때, batch_size 가 500이면 1번의 epoch 동안
2번의 iteration 이 발생
'''
