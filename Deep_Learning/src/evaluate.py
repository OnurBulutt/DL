import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(_, _),(x_test,y_test)=cifar10.load_data()
x_test=x_test/255.0
y_test=to_categorical(y_test,10)

model=tf.keras.models.load_model('cnn_cifar10_model.h5')
loss,acc=model.evaluate(x_test,y_test)
print('Accuracy:',acc)
