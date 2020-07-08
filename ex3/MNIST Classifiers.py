import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten,  MaxPooling2D, Conv2D, Input, UpSampling2D, Reshape
from keras import optimizers
from keras import backend as K
from keras.utils import np_utils
from sklearn.decomposition import PCA

def linear_model(image_train, label_train, image_test, label_test):
    model = Sequential()
    model.add(Dense(10, input_shape=(784,)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(image_train, label_train,
          batch_size=128, epochs=20,
          validation_data=(image_test, label_test))
    return history

def mlp(image_train, label_train, image_test, label_test):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(image_train, label_train,
          batch_size=128, epochs=20,
          validation_data=(image_test, label_test))
    return history

def convnet(image_train, label_train, image_test, label_test, batch_size=128, epochs=20, lr=0):
    image_train = image_train.reshape(60000,28,28,1)
    image_test = image_test.reshape(10000,28,28,1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)) )
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())          
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    if lr!=0:
        sgd = optimizers.SGD(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=sgd , metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(image_train, label_train, batch_size=batch_size, epochs=epochs,
          validation_data=(image_test, label_test))
    return history

def autoencoder(image_train, image_test):
    image_train = image_train.reshape((60000, 28, 28 ,1))
    image_test = image_test.reshape((10000, 28, 28 ,1))
    input_img = Input(shape=(28, 28, 1)) 

    x = Flatten()(input_img)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    encoded = Dense(2, activation='relu', name='encoder')(x)
    
    x = Dense(128, activation='relu')(encoded)
    x = Dense(512, activation='relu')(x)
    decoded = Dense(784, activation='sigmoid')(x)
    
    autoencoder = Model(input_img, Reshape((28, 28, 1))(decoded))
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = autoencoder.fit(image_train, image_train,
                epochs=30,
                batch_size=128,
                shuffle= True,
                
                validation_data=(image_test, image_test))
    
    return history, autoencoder
def plot_figures(history, model):
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title("epoch vs accuracy for " + model)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig("epoch vs accuracy for " + model)
    
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("epoch vs loss for " + model)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig("epoch vs loss for " + model)
    
if __name__ == '__main__':    
    (image_train, label_train), (image_test, label_test) = mnist.load_data()
    image_train = image_train.reshape((60000, 784))
    image_test = image_test.reshape((10000, 784))
    image_train = image_train.astype('float64') / 255
    image_test = image_test.astype('float64') / 255
    num_classes = 10
    
    label_train_categorical = np_utils.to_categorical(label_train, num_classes)
    label_test_categorical =  np_utils.to_categorical(label_test, num_classes)
    
    history_linear = linear_model(image_train, label_train_categorical, image_test, label_test_categorical)
    plot_figures(history_linear, "linear model")
    
    history_mlp = mlp(image_train, label_train_categorical, image_test, label_test_categorical)
    plot_figures(history_mlp, "mlp model")
    
    history_convnet = convnet(image_train, label_train_categorical, image_test, label_test_categorical)
    plot_figures(history_convnet, "convnet model")
    
    history_convnet_lr_0_0001 = convnet(image_train, label_train_categorical, image_test, label_test_categorical, epochs=5,lr=0.0001)
    history_convnet_lr_0_001  = convnet(image_train, label_train_categorical, image_test, label_test_categorical, epochs=5,lr=0.001)
    history_convnet_lr_0_01  = convnet(image_train, label_train_categorical, image_test, label_test_categorical, epochs=5,lr=0.01)
    history_convnet_lr_0_1  = convnet(image_train, label_train_categorical, image_test, label_test_categorical, epochs=5,lr=0.1)
    history_convnet_lr_1  = convnet(image_train, label_train_categorical, image_test, label_test_categorical, epochs=5,lr=1)
    for param in ['acc', 'val_acc', 'loss', 'val_loss']:
        fig = plt.figure()
        plt.plot(history_convnet_lr_0_0001.history[param])
        plt.plot(history_convnet_lr_0_001.history[param])
        plt.plot(history_convnet_lr_0_01.history[param])
        plt.plot(history_convnet_lr_0_1.history[param])
        plt.plot(history_convnet_lr_1.history[param])
        plt.title("epoch vs " + param + " for different learning rates on train data")
        plt.ylabel(param)
        plt.xlabel('epoch')
        plt.legend(['learning rate = 0.0001', 'learning rate = 0.001', 'learning rate = 0.01', 'learning rate = 0.1','learning rate = 1'], loc='best')
        plt.savefig("epoch vs " + param + " for different learning rates on train data")

    autoencoder_history, autoencoder = autoencoder(image_train, image_test)
    plot_figures(autoencoder_history, "autoencoder")
    
    get_encoder_output = K.function([autoencoder.layers[0].input],
                                  [autoencoder.get_layer("encoder").output])
    idx = np.random.choice(60000,5000, replace=False)
    
    encoder_output = np.array(get_encoder_output([image_train[idx]])).reshape((5000,2))

    fig = plt.figure()
    plt.scatter(encoder_output[:,0],encoder_output[:,1], c =label_train[idx] ,cmap='rainbow')
    plt.colorbar()
    plt.title("autoencoder scatter")
    plt.savefig("autoencoder scatter")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(image_train[idx])

    
    fig = plt.figure()
    plt.scatter(principal_components[:,0],principal_components[:,1], c =label_train[idx] ,cmap='rainbow')
    plt.colorbar()
    plt.title("PCA scatter")
    plt.savefig("PCA scatter")
    
    image_test = image_test.reshape((10000, 28, 28 ,1))
    decoded_imgs = autoencoder.predict(image_test)
    n = num_classes
    plt.figure(figsize=(40, 4))
    for i in range(1,n+1):
        # display original
        ax = plt.subplot(3, n, i)
        ax.set_title("original image", fontsize=20)
        plt.imshow(image_test[i-1].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        #display autoencoder image
        ax = plt.subplot(3, n, i + n)
        ax.set_title("autoencoder image", fontsize=20)
        plt.imshow(decoded_imgs[i-1].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display pca image
        ax = plt.subplot(3, n, i + 2*n)
        ax.set_title("pca image", fontsize=20)
        plt.imshow(pca.inverse_transform(pca.transform(image_test[i-1].reshape((1,-1)))).reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("images vs decoded images")
    