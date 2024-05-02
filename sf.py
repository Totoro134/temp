import secretflow as sf
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import os
from secretflow.utils.simulation.datasets import dataset
from secretflow.data.vertical import read_csv
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.utils.simulation.datasets import load_bank_marketing
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.preprocessing.encoder import LabelEncoder
from secretflow.data.split import train_test_split
from secretflow.security.privacy import DPStrategy, LabelDP
from secretflow.security.privacy.mechanism.tensorflow import GaussianEmbeddingDP
from secretflow.device.driver import reveal, wait

sf.init(['alice', 'bob', 'carol'], address='local')
alice, bob, carol = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('carol')

scaler = MinMaxScaler()

current_path = os.path.dirname(__file__)
x = read_csv({alice: os.path.join(current_path,'data/x1.csv'),
              bob: os.path.join(current_path,'data/x2.csv'),
              carol: os.path.join(current_path,'data/x3.csv')}, no_header = True)
x = scaler.fit_transform(x)
y = read_csv({alice: os.path.join(current_path,'data/y.csv')}, no_header = True)
x_test = read_csv({alice: os.path.join(current_path,'data/x_test1.csv'),
                   bob: os.path.join(current_path,'data/x_test2.csv'),
                   carol: os.path.join(current_path,'data/x_test3.csv')}, no_header = True)
x_test = scaler.fit_transform(x_test)
y_test = read_csv({alice: os.path.join(current_path,'data/y_test.csv')}, no_header = True)

# print('x1 = ')
# print(reveal(x.partitions[alice].data))
# print('x2 = ')
# print(reveal(x.partitions[bob].data))
# print('x3 = ')
# print(reveal(x.partitions[carol].data))
# print('y = ')
# print(reveal(y.partitions[alice].data))
# print('x_test = ')
# print(reveal(x_test.partitions[alice].data))
# print('y_test = ')
# print(reveal(y_test.partitions[alice].data))

spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))

def create_base_model(input_dim, output_dim, name='base_model'):
    # Create model
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf

        model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Dense(100, activation="relu"),
                layers.Dense(output_dim, activation="relu"),
            ]
        )
        # Compile model
        model.summary()
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_model

hidden = 64

model_base_alice = create_base_model(3, hidden)
model_base_bob = create_base_model(2, hidden)
model_base_carol = create_base_model(2, hidden)

def create_fuse_model(input_dims, output_dim, name='fuse_model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf

        # input
        input_layers = []
        for input_dim in input_dims:
            input_layers.append(
                keras.Input(
                    input_dim,
                )
            )

        merged_layer = layers.concatenate(input_layers)
        fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
        output = layers.Dense(output_dim, activation='sigmoid')(fuse_layer)

        model = keras.Model(inputs=input_layers, outputs=output)
        model.summary()

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_model


model_fuse = create_fuse_model([hidden, hidden, hidden], output_dim=1)

base_model_dict = {alice: model_base_alice, bob: model_base_bob, carol: model_base_carol}

# Define DP operations
train_batch_size = 600
gaussian_embedding_dp = GaussianEmbeddingDP(
    noise_multiplier=0.5,
    l2_norm_clip=1.0,
    batch_size=train_batch_size,
    num_samples=x.values.partition_shape()[alice][0],
    is_secure_generator=False,
)
label_dp = LabelDP(eps=64.0)
dp_strategy_alice = DPStrategy(label_dp=label_dp)
dp_strategy_bob = DPStrategy(embedding_dp=gaussian_embedding_dp)
dp_strategy_carol = DPStrategy(embedding_dp=gaussian_embedding_dp)
dp_strategy_dict = {alice: dp_strategy_alice, bob: dp_strategy_bob, carol: dp_strategy_carol}
dp_spent_step_freq = 10

sl_model = SLModel(
    base_model_dict=base_model_dict,
    device_y=alice,
    model_fuse=model_fuse,
    dp_strategy_dict=dp_strategy_dict,
)
st = time.time()
history = sl_model.fit(
    x,
    y,
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=train_batch_size,
    shuffle=False,
    verbose=1,
    validation_freq=1,
    dp_spent_step_freq=dp_spent_step_freq,
)
ed = time.time()
print(f'all = {ed - st}')
print(f'avg = {(ed - st)/20}')

# Plot the change of loss during training
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Plot the change of accuracy during training
plt.plot(history['train_accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot the Area Under Curve(AUC) of loss during training
plt.plot(history['train_auc_1'])
plt.plot(history['val_auc_1'])
plt.title('Model Area Under Curve')
plt.ylabel('Area Under Curve')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

global_metric = sl_model.evaluate(x_test, y_test, batch_size=train_batch_size)

Y_predict = sl_model.predict(x_test)
# print(Y_predict)
# print(reveal(Y_predict))
# Y_predict.to_csv('sf.csv')

Y_predict = reveal(Y_predict)
res = np.vstack(Y_predict)
np.savetxt('sf.csv', res, delimiter=',')