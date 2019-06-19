
from TensorANFIS.anfis import ANFIS
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt


# Dataset
dataframe = pd.read_csv('iris.csv', header=None)
dataset = dataframe.values

# Inputs
X = dataset[:,0:4]

# Labels
Y = dataset[:,4:6] # Note that the corresponding labels and index is at
                   # Y's first and second colum respectively
seed = 9
# Split in train and validation sets
X_train, X_val,Y_train, Y_val = train_test_split(X,Y,
                                                test_size=0.2,
                                                random_state=seed)
# Anfis class
D          = 4     # Nº of inputs
m          = 16    # Rules
alpha      =  0.01 # Training rate
num_epochs = 600   # Nº of epochs
fis = ANFIS(n_inputs=D, n_rules=m, learning_rate=alpha)

# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    # Initialize model parameters
    sess.run(fis.init_variables)
    trn_costs = []
    val_costs = []
    time_start = time.time()
    for epoch in range(num_epochs):
        #  Run an update step
        trn_loss, trn_pred = fis.train(sess, X_train, Y_train[:,0])
        # Evaluate on validation set
        val_pred, val_loss = fis.infer(sess, X_val, Y_val[:,0])
        if epoch % 10 == 0:
            print("Train cost after epoch %i: %f" % (epoch, trn_loss))
        if epoch == num_epochs - 1:
            time_end = time.time()
            print("Elapsed time: %f" % (time_end - time_start))
            print("Validation loss: %f" % val_loss)
            # Plot real vs. predicted
            pred = np.vstack((np.expand_dims(val_pred, 1)))
            pred = np.round(pred)
        trn_costs.append(trn_loss)
        val_costs.append(val_loss)

    plt.figure(1)
    plt.plot(Y_val[:,0],'bv',label="True class")
    plt.plot(pred,'r^',label="Predicted class")
    plt.xlabel("Dataset sample index")
    plt.ylabel("Corresponding class")
    plt.legend()
    plt.grid()
    # Plot the cost over epochs
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(np.squeeze(trn_costs))
    plt.title("Training loss, Learning rate =" + str(alpha))
    plt.subplot(2, 1, 2)
    plt.plot(np.squeeze(val_costs))
    plt.title("Validation loss, Learning rate =" + str(alpha))
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    # Plot resulting membership functions
    fis.plotmfs(sess)
    plt.show()
