from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from model import create_cnn_model

(x_train,y_train),_ = cifar10.load_data()
x_train = x_train/255.0
y_train = to_categorical(y_train,10)

model = create_cnn_model()
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,batch_size=32,validation_split=0.2)
