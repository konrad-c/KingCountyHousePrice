
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
from keras.optimizers import RMSprop, Adam
from keras import metrics

# my modules
import plot_helper as plotter
import data_handler
import house_price_cluster_analysis as cluster_classifier

classification_model = cluster_classifier.cluster_classification()

def classify_cluster(x_train, x_valid, x_test):
    #classification_model = cluster_classifier.cluster_classification()
    x_train_clusters = classification_model.predict(np.array(x_train))
    x_valid_clusters = classification_model.predict(np.array(x_valid))
    x_test_clusters = classification_model.predict(np.array(x_test))
    for col in range(x_train_clusters.shape[1]):
        colname = "cluster" + str(col)
        x_train[colname] = x_train_clusters[:,col]
        x_valid[colname] = x_valid_clusters[:,col]
        x_test[colname] = x_test_clusters[:,col]
    #x_train["cluster"] = np.argmax(x_train_clusters, axis=1)
    #x_valid["cluster"] = np.argmax(x_valid_clusters, axis=1)
    #x_test["cluster"] = np.argmax(x_test_clusters, axis=1)
    return (x_train, x_valid, x_test)

def price_prediction_model(input_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(input_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(20, activation="relu"))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(1))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)
    
def price_prediction_model_deep(input_size):
    t_model = Sequential()
    t_model.add(Dense(50, activation="tanh", input_shape=(input_size,), kernel_initializer="normal"))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(30, activation="relu", kernel_initializer="normal"))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(20, activation="relu", kernel_initializer="normal"))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(10, activation="relu", kernel_initializer="normal"))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(1))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)

def price_prediction():
    x_train, x_valid, x_test, y_train, y_valid, y_test = data_handler.train_valid_test_split_clean(method="zscore")
    
    #x_train, x_valid, x_test = classify_cluster(x_train, x_valid, x_test)
    
    # Hyperparameters
    batch_size = 32
    epochs = 500

    # Get train set
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    # Get valid set
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    
    # Get test set
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # Train model
    input_size = x_train.shape[1]
    np.random.seed(2017)
    model = price_prediction_model_deep(input_size)
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    shuffle=True,
                    validation_data=(x_valid, y_valid))
    score = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, batch_size=128)
    
    # Investigate Nature of Errors
    plotter.plot_error(predictions, y_test)
    print("Correlation of predictions and real:", np.corrcoef(predictions[:, 0], y_test)[0,1]**2, "\n",
            "Test MSE:", score[0], "\n",
            "Test MAE:", score[1])
    return model
    
    


