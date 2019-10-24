import tensorflow as tf
from tqdm import tqdm
import os 

class tfRecord_Processor():

  def __init__(self):
    pass

  def read_raw(self, file, doPrint = False):
    tfrecords = []
    for example in tf.python_io.tf_record_iterator(file):
      tf_example = tf.train.SequenceExample.FromString(example)
      if doPrint:
        print(example)
      else:
        tfrecords.append(tf_example)
    if not doPrint:
      return tfrecords

  def clip_write_directory(self, inDir, outDir, keep_labels):
     """ inDir: directory containing .tfrecord files 
         outDir: directory where new .tfrecord files will be saved 
         use for: clip and write each example in each .tfrecord file in 'inDir'
         see clip_write() for rest of detials 
    """
    dirFiles = os.listdir(inDir)
    with tqdm(total = len(dirFiles)) as pbar: 
      for file in dirFiles:
        if '.tfrecord' in file:
          tfrecord = os.path.join(inDir, file)
          outfile = os.path.join(outDir, 'clipped' + file)
          print('a')
          with tf.python_io.TFRecordWriter(outfile) as tfwriter:
            print('b')
            # iterate through all examples in tfrecord 
            for example in tf.python_io.tf_record_iterator(tfrecord):
              tf_example = tf.train.SequenceExample.FromString(example)
              labels = tf_example.context.feature['labels'].int64_list.value
              if any(l in labels for l in keep_labels):
                clipped_tf_example = self.__clip_labels(tf_example, keep_labels)
                tfwriter.write(clipped_tf_example.SerializeToString())
            # tfwriter.close()
            # sys.stdout.flush()
        pbar.update(1)

  def clip_write(self, infile, keep_labels, outfile):
    """ read t.tfrecords
        select those records with labels in 'keep_labels'
        remove from those records the labels not in 'keep_labels'
        write to 'outfile'
    """
    writer = tf.python_io.TFRecordWriter(outfile)
    for example in tf.python_io.tf_record_iterator(infile):
        tf_example = tf.train.SequenceExample.FromString(example)

        labels = tf_example.context.feature['labels'].int64_list.value
        if any(l in labels for l in keep_labels):
          new_tf_example = self.__clip_labels(tf_example, keep_labels)
          writer.write(new_tf_example.SerializeToString())


  def __clip_labels(self, tf_example, keep_labels):
    """ modify 'labels' from context to include only labels in 'keep_labels'
        keep 'id' from context the same
        keep 'rgb' & 'audio' from feature_lists the same  
    """
    labels = tf_example.context.feature['labels'].int64_list.value
    new_labels = [l for l in labels if l in keep_labels]

    new_context = tf.train.Features(feature={
    'id': tf_example.context.feature['id'],
    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=new_labels))
    })

    new_example = tf.train.SequenceExample(
      context = new_context,
      feature_lists = tf_example.feature_lists)

    return new_example