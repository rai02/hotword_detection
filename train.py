import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


DATA_PATH = "data.json"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1
NUM_KEYWORDS = 11

SAVED_MODEL_PATH="model.h5"

def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    #extract input & target
    X = np.array(data["MFCCs"])
    Y = np.array(data["labels"])
    return X,Y



def get_data_splits(data_path):
    #load dataset
    X, Y  = load_dataset(data_path)
    #create train, valid, test splits
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= TEST_SIZE)
    X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=VALIDATION_SIZE)

    #convert from 2d to 3d array
    X_train = X_train[...,np.newaxis]
    X_validation = X_validation[...,np.newaxis]
    X_test = X_test[...,np.newaxis]
    return X_train,X_validation,X_test,Y_train,Y_validation,Y_test

def build_model(input_shape, lr, error="sparse_categorical_crossentropy"):
    # build network
    model = keras.Sequential()
    # 3 conv laver
    # conv 1
    #keras.layers.Conv2d(filter size,kernel dim, activation, input, shape)
    model.add(keras.layers.Conv2D(64, (3, 3),activation="relu", input_shape=input_shape,
              kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3,3),strides=(2,2),padding="same"))
    # conv 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
              kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3,3),strides=(2,2),padding="same"))

    # conv 3
    model.add(keras.layers.Conv2D(64, (2, 2), activation="relu",
              kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2,2),strides=(2,2),padding="same"))

    # flatten output -> feed it to  dense layer 3d TO 1D  AS DENSE LAYER NEED 1D INPUT
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # softmax classifier
    # model.add(keras.layers.Dense(NUM_KEYWORDS=NUM_KEYWORDS,activation="softmax"))
    model.add(keras.layers.Dense(11, activation='softmax'))

    #compile the model
    opttimiser = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opttimiser,loss=error, metrics=["accuracy"])

    # print model details
    model.summary()

    # return model
    return model


def main():
    #load trianing/validation/test splits
    X_train, X_validation, X_test, Y_train,Y_validation,Y_test = get_data_splits(DATA_PATH)
    print(X_train[0],X_train[0].shape)
    #build CNN model
    input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]) # (# segments, no of coeff, 1-> fundamental for cnn depth or channel for image (just like an image for audio)
    # print(input_shape)
    model = build_model(input_shape, LEARNING_RATE)
    #train model
    model.fit(X_train,Y_train,epochs=EPOCHS,batch_size = BATCH_SIZE,validation_data = (X_validation,Y_validation))

    #evaluate model
    test_errror, test_accuracy = model.evaluate(X_test,Y_test)
    print(f"test_error:{test_errror}, test_accuracy: {test_accuracy}")
    #save the model
    model.save(SAVED_MODEL_PATH)

if __name__ == "__main__":
    main()