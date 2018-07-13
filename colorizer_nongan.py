import os
import keras
from keras.layers import Conv2D ,UpSampling2D,Input,BatchNormalization, Activation, Dropout,Concatenate,MaxPooling2D  
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import LeakyReLU
import utils
import numpy as np
from keras.callbacks import ModelCheckpoint,CSVLogger,TensorBoard
from subpixel import SubpixelConv2D
import keras.backend as K
import argparse

print(K.image_dim_ordering())

## Модель глубокой нейронной сети на базе автоэнкодера.
## Попытка обучить нейронную сеть способную повышать пространственное разрешение
## спутниковых изображений.

def model_multi_input():
    """
    Подготовка объекта класса Model, содержащего модель нейронной сети.
    """
    inputs = Input(shape=(128,128,3))

    conv1 = Conv2D(32, kernel_size=(3,3), padding='same')(inputs)
    conv1 = BatchNormalization(momentum=0.8)(conv1)
    conv1 = (LeakyReLU(0.2))(conv1)

    conv1 = Conv2D(32, kernel_size=(3,3), padding='same')(conv1)
    conv1 = BatchNormalization(momentum=0.8)(conv1)
    conv1 = (LeakyReLU(0.2))(conv1)

    conv2 = Conv2D(64, kernel_size=(3,3), padding='same')(conv1)
    conv2 = BatchNormalization(momentum=0.8)(conv2)
    conv2 = (LeakyReLU(0.2))(conv2)

    conv2 = Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same')(conv2)
    conv2 = BatchNormalization(momentum=0.8)(conv2)
    conv2 = (LeakyReLU(0.2))(conv2)

    conv3 = Conv2D(128, kernel_size=(3,3), padding='same')(conv2)
    conv3 = BatchNormalization(momentum=0.8)(conv3)
    conv3 = (LeakyReLU(0.2))(conv3)

    conv3 = Conv2D(128, kernel_size=(3,3), strides=(2,2),padding='same')(conv3)
    conv3 = BatchNormalization(momentum=0.8)(conv3)
    conv3 = (LeakyReLU(0.2))(conv3)

    conv4 = Conv2D(256, kernel_size=(3,3), padding='same')(conv3)
    conv4 = BatchNormalization(momentum=0.8)(conv4)
    conv4 = (LeakyReLU(0.2))(conv4)

    conv4 = Conv2D(256, kernel_size=(3,3), strides=(2,2), padding='same')(conv4)
    conv4 = BatchNormalization(momentum=0.8)(conv4)
    conv4 = (LeakyReLU(0.2))(conv4)

    up7 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv4), conv3])
    conv7 = Conv2D(128, kernel_size=(3,3), padding='same')(up7)
    conv7 = BatchNormalization(momentum=0.8)(conv7)
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(128, kernel_size=(3,3), padding='same')(conv7)
    conv7 = BatchNormalization(momentum=0.8)(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, kernel_size=(3,3), padding='same')(up8)
    conv8 = BatchNormalization(momentum=0.8)(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(64, kernel_size=(3,3), padding='same')(conv8)
    conv8 = BatchNormalization(momentum=0.8)(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, kernel_size=(3,3), padding='same')(up9)
    conv9 = BatchNormalization(momentum=0.8)(conv9)
    conv9 = Activation('relu')(conv9)

    conv9 = Conv2D(32, kernel_size=(3,3), padding='same')(conv9)
    conv9 = BatchNormalization(momentum=0.8)(conv9)
    conv9 = Activation('relu')(conv9)
    conv10 = Conv2D(1,1,activation='sigmoid')(UpSampling2D(size=(2, 2))(conv9))
    model = Model(input=inputs, output=conv10)

    print('model compile')
    return model


def main(args):
    """Основная функция, выполняющая обучение модели."""
    # Количество эпох обучения.
    num_steps = 1000
    # Размер батча.
    batch_size = 1
    colorizer = model_multi_input()

    if os.path.exists(args.model_path):
        colorizer.load_weights(args.model_path)

    print(colorizer.summary())
    colorizer.compile(optimizer=Adam(lr=2e-4,beta_1=0.5), loss='binary_crossentropy', metrics=['mse','mae','accuracy'])
    
    # Подготовка батчей.
    igen = utils.batch_generator(path_x=args.data_path,batch_size=batch_size,max_steps=num_steps)
    valx,valy = utils.load_single_pair(args.data_path,-200,100) 
    valx = np.array(valx)
    valy = np.array(valy)
    vali = (valx,valy)

    print(np.histogram(valx))
    print(np.histogram(valy))
    # Настройка процесса обучения. Сохранение чекпоинтов, написание логов, "живая" визуалищация.
    checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss', verbose=1, save_best_only=True)
    logger = CSVLogger(filename=args.model_path.replace('.mdl','.csv'), append=False)
    tboard = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=1, write_graph=True, write_grads=True, write_images=True)

    colorizer.fit_generator(igen,validation_data=vali, steps_per_epoch=250, epochs=10, verbose=1, callbacks=[checkpoint,logger,tboard],
                  max_queue_size=10,shuffle=True, initial_epoch=0,use_multiprocessing=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    main(args)
