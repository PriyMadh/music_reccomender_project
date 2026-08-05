import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load data 
df = pd.read_csv("ML_Projects/music_recomender_system_data.csv", index_col = 0)

# Y matrix - actual ratings
Y_train = df.fillna(0).to_numpy()

# R matrix (yes/no rated)
R_train = df.notna().astype(int).to_numpy()

print("Y (Ratings Matrix):\n", Y_train)
print("\nR (Indicator Matrix):\n", R_train)


def cost_function(X, W, b, Y, R, lambda_):
    ns, nu = Y.shape # number of songs, number of users
    J = 0 # initialize cost
    for j in range (nu):
        w_j = W[j] # different wieght vector for each user
        b_j = b[0, j] # diff bias for each user
        for i in range (ns): #run this for each user for every song
            x_i = X[i]
            y_ij  = Y[i, j] # user rating
            r_ij =R[i, j] # was it rated or not
            prediction_ij = tf.tensordot(w_j, x_i, axes=1) + b_j #have to use tensorlfow functions so that Gradient tape doesn't lose track of the math
            err = (prediction_ij - y_ij) * r_ij
            J += tf.square(err)
            
    J = J / 2.0
    
    # Regularization
    reg_W = (lambda_ / 2.0) * tf.reduce_sum(tf.square(W))
    reg_X = (lambda_ / 2.0) * tf.reduce_sum(tf.square(X))
    
    return J + reg_W + reg_X

def normalize_data(Y, R): # mean normalizatoin of data

    num_songs = Y.shape[0]
    Ymean = np.zeros((num_songs, 1))
    Ynorm = np.zeros(Y.shape)
    
    for i in range(num_songs):
        idx = (R[i, :] == 1) # make sure to take the average of the ones that are rated
        if np.any(idx):
            Ymean[i] = np.mean(Y[i, idx])
            Ynorm[i, idx] = Y[i, idx] - Ymean[i]
        else:
            Ymean[i] = 0
            
    return Ynorm, Ymean

Ynorm, Ymean = normalize_data(Y_train, R_train)

num_songs, num_users  = Y_train.shape
num_features = 5 # arbitraryish, but like I just wanted one feature to coorespond with one distinct genre of musics so that's why I picked this
tf.random.set_seed(1234)
W = tf.Variable(tf.random.normal((num_users, num_features)), name  = 'W') # create W matrix to be optimized
X = tf.Variable(tf.random.normal((num_songs, num_features)), name = "X") # X matrix to be optimized
b = tf.Variable(tf.random.normal((1, num_users)), name = 'b') # b vector to be optimized

optimizer = keras.optimizers.Adam(learning_rate = 1e-1)
iterations = 150
lambda_ = 1

Ynorm_tensor = tf.constant(Ynorm, dtype=tf.float32) # convert to varaibales types compatable wiht tensorflow
R_tensor = tf.constant(R_train, dtype=tf.float32)

for iter in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cost_function(X, W, b, Ynorm_tensor, R_tensor, lambda_)
    gradients = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(gradients, [X, W, b]))
    if iter % 10 == 0:
        print(f"cost is {cost_value}")
    

predictions = np.zeros((num_songs, num_users))

for i in range (num_songs):
    x_i = X[i]
    for j in range (num_users):
        w_j = W[j]
        b_j = b[0, j]
        predictions[i, j] = (np.dot(x_i, w_j) + b_j) + Ymean[i,0]
print(f"prediction matrix: {predictions}")
print(f"feature vectors for songs: {X}")
print(f"weights for users: {W}")








