
# standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SKLearn Clustering imports
from sklearn.cluster import KMeans

# Keras NN modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras import metrics

# my modules
import plot_helper as plotter
import data_handler

# Cluster 
def cluster_df():
    x_train, x_valid, x_test, y_train, y_valid, y_test = data_handler.train_valid_test_split_clean(method="normal")
    x_train['price'] = y_train
    x_valid['price'] = y_valid
    cluster_frame = pd.concat([x_train, x_valid])
    
    # Perform KMeans clustering
    kmeans_model = KMeans(n_clusters=3).fit(cluster_frame)
    cluster_frame["cluster"] = kmeans_model.labels_
    # Remove PRICE from input data
    cluster_frame = cluster_frame.drop("price", axis=1)
    train = cluster_frame.iloc[0:x_train.shape[0], :]
    valid = cluster_frame.iloc[x_train.shape[0]:, :]
    return train, valid

def cluster_classification_model():
    train, valid = cluster_df()
    input_size = train.shape[1] - 1
    t_model = Sequential()
    t_model.add(Dense(90, activation="relu", input_shape=(input_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(3, activation="softmax"))
    print(t_model.summary())
    t_model.compile(loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=["accuracy"])
    return(t_model)

def cluster_classification():
    train, valid = cluster_df()
    batch_size = 64
    epochs = 30

    # Get train set
    x_train = np.array(train.drop("cluster", axis=1))
    y_train = np.array(train.cluster)
    
    # Get valid set
    x_valid = np.array(valid.drop("cluster", axis=1))
    y_valid = np.array(valid.cluster)
    
    # Convert target to categorical:
    y_train = keras.utils.to_categorical(y_train)
    y_valid = keras.utils.to_categorical(y_valid)
    
    # Train model
    model = cluster_classification_model()
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_valid, y_valid))
    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model
    
    





