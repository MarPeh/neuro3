import pandas as pd
from datetime import timedelta
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import pickle
from typing import List
# Читаем файл с данными, в последствии будем выбирать из API
df = pd.read_csv("data_full-5.csv", index_col='Unnamed: 0')
# Печатаем форму и содержимое
print(df.shape)
print(df.head())
# Ну и заодно смотрим статистику
print(df["Close"].min(), df["Close"].max())

# График закрытия
#plt.plot(df.index,df["Open"])
#plt.show()
# посмотрим на связь между значениями записи, если корелляция сильная - лишнее брать не будем
print(df[["Open", "Close", "Low", "High", "Volume", "PriceOnDay5"]].corr())
# Отбираем 60% в выборку для обучения, остальное будет для валидации и проверки модели
train_size = int(len(df.index)*0.6)
print(train_size)

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

scaler = StandardScaler()
scaler.fit(train_df[["Low"]])
pickle.dump(scaler, open('scal','wb'))

def make_dataset(
        df,
        window_size,
        batch_size,
        use_scaler=True,
        shuffle=True,
        sequence_stride = 2
        ):
      features = df[["Low"]].iloc[:-window_size]
      if use_scaler:
        features = scaler.transform(features)
      data = np.array(features, dtype=np.float32)
      ds = tf.keras.preprocessing.timeseries_dataset_from_array(
          data=data,
          targets=df["Low"].iloc[window_size:],
          sequence_length=window_size,
          sequence_stride=sequence_stride,#Период между последовательными выходными последовательностями.
                            # Для шага s выходные образцы будут начинаться с индекса data[i] , data[i + s] , data[i + 2 * s] и т. Д.
                            # свечей вперед
          shuffle=shuffle,
          batch_size=batch_size)
      return ds

example_ds = make_dataset(df=train_df, window_size=3, batch_size=2, use_scaler=False, shuffle=False)

example_feature, example_label = next(example_ds.as_numpy_iterator())

print(example_feature.shape)
print(example_feature)

print(example_label.shape)
print(train_df["Low"].iloc[:6])

print(example_feature[0])
print(example_label[0])

print(example_feature[1])
print(example_label[1])

window_size = 100
batch_size = 16
sequence_stride = 2
train_ds = make_dataset(df=train_df, window_size=window_size, batch_size=batch_size, use_scaler=True, shuffle=True,sequence_stride=sequence_stride)
val_ds = make_dataset(df=val_df, window_size=window_size, batch_size=batch_size, use_scaler=True, shuffle=True, sequence_stride = sequence_stride)


# lstm_model = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(100, return_sequences=True),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.LSTM(50, return_sequences=True),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.LSTM(25, return_sequences=False),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(1)
# ])

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.Dense(1)
])






def compile_and_fit(model, train_ds, val_ds, num_epochs: int = 20):
  model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()]
      )
  history = model.fit(
      train_ds,
      epochs=num_epochs,
      validation_data=val_ds,
      verbose=1#,
      #batch_size=256
      )
  return history


history =  compile_and_fit(lstm_model, train_ds, val_ds, num_epochs=30)
lstm_model.save('model4')
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.show()
exit()
lstm_model.evaluate(train_ds)

lstm_model.evaluate(val_ds)
exit()
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(1)

])

history =  compile_and_fit(lstm_model, train_ds, val_ds, num_epochs=500)

history =  compile_and_fit(lstm_model, train_ds, val_ds, num_epochs=100)

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.show()


lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

history =  compile_and_fit(lstm_model, train_ds, val_ds, num_epochs=500)

history =  compile_and_fit(lstm_model, train_ds, val_ds, num_epochs=500)

history =  compile_and_fit(lstm_model, train_ds, val_ds, num_epochs=100)

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.show()
