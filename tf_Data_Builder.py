import tensorflow as tf
from tqdm import tqdm
import os 
import sys

class tf_Data_Builder():

  def __init__(self, batchsize):
    self.batchsize = batchsize

  def create_dataset(self, tf_files):

    dataset = tf.data.TFRecordDataset(tf_files)
    dataset = dataset.map(self.__parse_function, num_parallel_calls=4)
    dataset = dataset.repeat()    
    dataset = dataset.shuffle(len(tf_files))
    dataset = dataset.batch(self.batchsize)

    iterator = dataset.make_one_shot_iterator()
    ID, label, image, audio = iterator.get_next()

    sess = tf.InteractiveSession()
    image = tf.decode_raw(image.values, out_type = tf.uint8)
    image = tf.reshape(image, [self.batchsize,300,1])
    audio = tf.decode_raw(audio.values, out_type = tf.uint8)
    audio = tf.reshape(audio, [self.batchsize,300,1])

    print('ID shape: {}'.format(tf.shape(ID).eval()))
    print('Labels shape: {}'.format(tf.shape(label).eval()))
    print('Images shape: {}'.format(tf.shape(image).eval()))
    print('Audio shape: {}'.format(tf.shape(audio).eval()))
    sess.close()

    return ID, label, image, audio

  def __parse_function(self, raw_tfrecord):

    context_features = {
        'id': tf.FixedLenFeature([], dtype=tf.string),
        'labels': tf.VarLenFeature(dtype=tf.int64)
    }
    sequence_features = {
        'audio': tf.VarLenFeature(dtype=tf.string),
        'rgb': tf.VarLenFeature(dtype=tf.string)
    }
    context_data, sequence_data = tf.parse_single_sequence_example(
      serialized = raw_tfrecord,
      context_features = context_features,
      sequence_features = sequence_features)

    ID = context_data['id']
    label = context_data['labels']
    audio = sequence_data['audio']
    image = sequence_data['rgb']

    return ID, label, image, audio
  