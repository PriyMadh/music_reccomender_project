import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load data 
df_1 = pd.read_csv("music_recomender_system_data 2.csv", index_col=0)
df_2 = pd.read_csv("music_content_based_filtering_song_data copy.csv", index_col=0)
df_3 = pd.read_csv("music_content_based_filtering_user_data 2.csv", skiprows=1, index_col=0)

df_users_encoded = pd.get_dummies(df_3, drop_first=True).astype(np.float32)
df_songs_encoded = pd.get_dummies(df_2, drop_first=True).astype(np.float32)

# Y matrix - actual ratings
Y = df_1.fillna(0).to_numpy()

# R matrix (yes/no rated)
R = (Y > 0).astype(int)

song_indices, user_indices = np.where(R == 1) # finds the indices of R matrix where R = 1 (where Y is rated) (outputs two different 1-D list (for example if there are ratings at (0,0), (0,1), and (2, 3), the two list would be [0,0,2] and [0, 1, 3] and then later zip() can zip them back coorespondingly))

# User data
X_user = df_users_encoded.to_numpy()

# Song data
X_song = df_songs_encoded.to_numpy()

# use only the data where R = 1
# this unrolls the data so that we have pairs of X_song and X_user for each training example even if we need some duplicates of one or the other (example multiple song 1 to coorespond to multile people who rated song 1)

X_user_train = X_user[user_indices]
X_song_train = X_song[song_indices]
Y_train = Y[user_indices, song_indices].astype(np.float32)


user_NN = tf.keras.models.Sequential([

    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dense(8)

]
)

song_NN = tf.keras.models.Sequential([

    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dense(8)

]
)

input_user = tf.keras.layers.Input(shape=(X_user_train.shape[1],))
input_song = tf.keras.layers.Input(shape=(X_song_train.shape[1],))

vu = user_NN(input_user)
vs = song_NN(input_song)

vu = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vu)
vs = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vs)

output = tf.keras.layers.Dot(axes = 1)([vu, vs])

model = tf.keras.Model(inputs=[input_user, input_song], outputs=output)
cost_fn = tf.keras.losses.MeanSquaredError()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError()
)

model.fit(
    x=[X_user_train, X_song_train], 
    y=Y_train, 
    epochs=50, 
    batch_size=2 #smaller batch size for smaller training data 
)