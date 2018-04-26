import tensorflow as tf

# read images from data
def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  key, value = reader.read(filename_queue)

  features = tf.parse_single_example(
      value,
      # Defaults are not specified since both keys are required.
      features={
        'labels': tf.FixedLenFeature([], tf.int64)
        #'mean_rgb': tf.FixedLenFeature([], tf.float),
        #'video_id': tf.FixedLenFeature([], tf.int64),
        #'mean_audio': tf.FixedLenFeature([], tf.float)
        })

  return features

def main(unused_argv):
  """Just reads data for now."""
  print("hi :-)")

  init_op = tf.global_variables_initializer()

  tfrecords_filename = 'features/trainW9.tfrecord'
  fname_q = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=1)

  features = read_and_decode(fname_q)

  with tf.Session()  as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    feats = sess.run(features)
    print("my feat", feats)

    coord.request_stop()
    coord.join(threads)

# Run
if __name__ == '__main__':
  tf.app.run()
