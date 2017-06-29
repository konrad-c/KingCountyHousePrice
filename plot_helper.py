import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_error(predictions, real):
    predictions = predictions.reshape(predictions.shape[0])
    squared_error = np.power(predictions-real, 2)
    absolute_error = np.abs(predictions-real)
    error_df = pd.DataFrame({
        "prices": real,
        "preds": predictions,
        "sq_error": squared_error,
        "abs_error": absolute_error
    })
    # Bin data:
    bins = np.linspace(error_df.preds.min(), error_df.preds.max(), 20)
    averages_abs = []
    averages_sq = []
    prices = np.array(error_df.prices)
    for i in range(1, len(bins)):
        mask = np.where(np.logical_and(prices > bins[i-1], prices <= bins[i]))
        averages_abs.append(np.mean(np.array(error_df.abs_error)[mask]))
        averages_sq.append(np.mean(np.array(error_df.sq_error)[mask]))
    
    figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,8))
    # Scatter Plot
    axes[0,0].plot(error_df.prices, error_df.preds, ".", alpha=0.2)
    axes[0,0].set_title("Real vs Prediction")
    axes[0,0].set_ylabel("Predicted")
    # Proportion Error 
    axes[0,1].plot(error_df.prices, error_df.abs_error/error_df.prices, ".", alpha=0.2)
    axes[0,1].set_title("Proportion of Absolute Error to Price")
    axes[0,1].set_ylabel("Proportion")
    # Square Error
    axes[1,0].plot(error_df.prices, error_df.sq_error, ".", alpha=0.2)
    axes[1,0].plot(bins[1:], averages_sq, "r--", lw=3)
    axes[1,0].set_title("Squared Error")
    # Absolute Error
    axes[1,1].plot(error_df.prices, error_df.abs_error, ".", alpha=0.2)
    axes[1,1].plot(bins[1:], averages_abs, "r--", lw=3)
    axes[1,1].set_title("Absolute Error")
    # Cumulative Error 
    axes[2,0].hist(error_df.prices, weights=error_df.abs_error, bins=40)
    axes[2,0].set_title("Cumulative Error")
    plt.tight_layout()
    plt.show()