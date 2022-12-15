# Databricks notebook source
import datetime
import os
import logging
import sys
import datetime
import glob
import time
import os
import datetime
import time
import pandas as pd 
import numpy as np
from pytz import timezone
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout,GlobalAveragePooling2D
import horovod.tensorflow.keras as hvd
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from sparkdl import HorovodRunner

# COMMAND ----------

curent_timestamp = datetime.datetime.now().astimezone(timezone('US/Pacific')).strftime("%Y_%m_%d_%H_%M_%S")
final_output_dir = "/dbfs/csci-653/horovod-mnist/logs/"
output_dir = 'logs/logs/horovod_logs/'
output_dir = output_dir + f"{curent_timestamp}/"
model_output_dir = output_dir + "model/"
for _path in [output_dir, final_output_dir, model_output_dir]:
  try:
    os.makedirs(_path)
    print(f"{_path} - file created successfully")
  except Exception as ex:
    if FileExistsError:
      print(f"{_path} - file already exist")
    else:
      raise ex

# COMMAND ----------

def save_model_single(filename):
  import shutil
  dest_dir = output_dir + str(np_setup) + "/" 
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
  shutil.copy(filename, dest_dir + filename)
  
def save_horovod_model():
  import shutil
  for path in glob.glob(checkpoint_dir+"/*"):
    desc_dir = model_output_dir + "NP" +str(np_setup) + "/"
    if not os.path.exists(desc_dir):
      os.makedirs(desc_dir)
    model_filename = os.path.basename(path)
    shutil.copy(path, desc_dir + model_filename)

# COMMAND ----------

batch_size = 128
epochs = 10
repeat = 1
learning_rate = 0.1
num_classes = 10

# COMMAND ----------

checkpoint_dir = '/dbfs/ml/MNISTDemo/train/{}/'.format(time.time())
os.makedirs(checkpoint_dir)

# COMMAND ----------

def get_dataset(num_classes, rank=0, size=1):
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('MNIST-data-%d' % rank)
  x_train = x_train[rank::size]
  y_train = y_train[rank::size]
  x_test = x_test[rank::size]
  y_test = y_test[rank::size]
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return (x_train, y_train), (x_test, y_test)

# COMMAND ----------

def get_model(num_classes):
  model = models.Sequential()
  model.add(layers.Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.25))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_classes, activation='softmax'))
  return model
def train(learning_rate=1.0):
  (x_train, y_train), (x_test, y_test) = get_dataset(num_classes)
  model = get_model(num_classes)

  optimizer = keras.optimizers.Adadelta(lr=learning_rate)

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))
  
  return model


# COMMAND ----------

env = "PROJECT"
instance_type = 'SINGLE_INSTANCE'
dataset = 'MNIST'
model_name = 'CNN'
(x_train, y_train), (x_test, y_test) = get_dataset(num_classes)
train_shape = x_train.shape[0]
validation_shape = x_test.shape[0]

def get_cluster_info(hvd_run=True):
  driver_type = sc.getConf().get("spark.databricks.driverNodeTypeId")
  worker_type = "None"
  num_workers = "None"
  if hvd_run == True:
    worker_type = sc.getConf().get("spark.databricks.workerNodeTypeId")
    num_workers = sc.getConf().get("spark.databricks.clusterUsageTags.clusterWorkers")
  return [driver_type, worker_type, num_workers]

driver_type, worker_type, num_workers = get_cluster_info()
np_setup = "SINGLE"

# COMMAND ----------

filename = f"benchmark_{env}_{dataset}_{model_name}_{train_shape}_{validation_shape}_{instance_type}_{driver_type}_{worker_type}_{num_workers}_{np_setup}_{repeat}_{epochs}_{learning_rate}_{batch_size}.log"

# COMMAND ----------



# COMMAND ----------

model = train()
model_file_name = filename.split(".log")[0] + ".h5"
model.save(model_file_name)
save_model_single(model_file_name)

# COMMAND ----------

def train_hvd(learning_rate = learning_rate):
  #STEP1: INITIALIZATION
  hvd.init()
  
  #DATA READ TIME
  start_time = time.time() 
  (x_train, y_train), (x_test, y_test) = get_dataset(num_classes, hvd.rank(), hvd.size())  
  end_time = round((time.time() - start_time),3)
  print(f"get_data_time - {end_time}")  

  start_time = time.time()
  model = get_model(num_classes)
  optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate * hvd.size())
  optimizer = hvd.DistributedOptimizer(optimizer)
  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  callbacks = [
      hvd.callbacks.BroadcastGlobalVariablesCallback(0),
      hvd.callbacks.MetricAverageCallback(),
  ]
  if hvd.rank() == 0:
      callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_dir + 'checkpoint-{epoch}.ckpt', save_weights_only = True))

  end_time = round((time.time() - start_time),3)
  print(f"step - prep_model - {end_time}") 
  
  # TRAINING STARTS HERE
  start_time = time.time()
  model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=callbacks,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test)
         )
   
  end_time = round((time.time() - start_time),3)
  print(f"step - train_model - {end_time}")   
    

# COMMAND ----------

instance_type = "HOROVOD_CLUSTER"
np_list = [1,2,4]
for np_setup in np_list:  
  start_time = time.time()
  
  checkpoint_dir = '/dbfs/ml/MNISTDemo/train/{}/{}/'.format(np_setup, time.time())
  os.makedirs(checkpoint_dir)
  
  filename = f"benchmark|{env}|{dataset}|{model_name}|{train_shape}|{validation_shape}|{instance_type}|{driver_type}|{worker_type}|{num_workers}|{np_setup}|{repeat}|{epochs}|{learning_rate}|{batch_size}.log"
  print(output_dir+filename)

  for i in range(repeat):
    print(f"{instance_type} - REPEAT {i+1}")
    hr = HorovodRunner(np = np_setup, driver_log_verbosity='all')
    hr.run(train_hvd)

  end_time = round((time.time() - start_time),3)
  print(f"np{np_setup} - finshed in {end_time}") 

# COMMAND ----------


