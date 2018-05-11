import numpy as np
import random
import tensorflow as tf
from retrain import * 
import sys
sys.path.append('./SoundNet-tensorflow')
from main import Model
from util import load_from_list

config = {
    'tfhub_module':'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1', # link to pretrained inception model
    'bottleneck_dir':'data/img_bottlenecks/', # for image feature representations
    'image_dir':'data/testImages_artphoto/', # 
    'summaries_dir':'logging/summaries/',
    'num_training_steps':1000000,
    'train_batch_size':32, # should be same as batch_size because flags are different for sound net and image net :)
    'eval_step_interval':1,
    'testing_percentage':10,
    'validation_percentage':10,
    'phase':'train',
    'param_g_dir':'./models/sound8.npy', # TODO: change to correct relative path (we r one dir up)
    'learning_rate':1e-3,
    'batch_size':32,
    'num_noise_samples':3,
    'hidden_dim':1024,
    'name_scope': 'SoundNet', # for things in sound net
    'eps':1e-5,
    'load_size':22050*4,
}

label_map = {'excitement':'joyful_activation',
             'awe':'amazement',
             'anger':'power',
             'amusement':'tenderness',
             'disgust':'nostalgia',
             'fear':'tension',
             'contentment':'calmness',
             'sad':'sadness'
            }

tf.logging.set_verbosity(tf.logging.INFO)

###################################
###          Utilities          ###
###################################

def NCE_sentiment_loss(img_embs, true_song_embs, noise_song_embs, epsilon=1e-5):
    """
    Args:
        img_embs: N x d
        true_song_embs: N x d
        noise_song_embs: N x (Noi * d)
    Returns: 
    """
    # compute softmax for each input
    label_logits = tf.einsum('ij,ij->i', img_embs, true_song_embs) # N
    noise_logits = tf.einsum('ij,igj->ig', img_embs, noise_song_embs) # N x Noi

    loss = tf.log(tf.sigmoid(label_logits)+epsilon) # N
    noise_loss = tf.log(tf.sigmoid(-noise_logits)+epsilon) # N x Noi

    # final expression for loss
    loss = - loss - tf.reduce_sum(noise_loss, axis=1) 
    loss = tf.reduce_mean(loss)
    return loss

def sample_song(labels, data_dir='./data/emotifymusic/'):
    """
    Args:
        labels: the classes of songs from which
        to sample (from image labels)
        data_dir: directory with song data
    Returns: song of type label, ready to be
        fed into network
    """
    song_files = []
    for label in labels:
        song_label = label_map[label]
        dir_name =  data_dir + song_label + '/'
        song_files.append(dir_name + random.choice(os.listdir(dir_name)))
    return load_from_list(song_files, config)

def sample_noise_songs(labels, k, data_dir='./data/emotifymusic/'):
    """sozz 4 research code
    Args:
        labels: list of TODO
    """
    song_choices = {}
    for emotion in label_map.values():
        dir_name =  data_dir + emotion + '/'
        song_choices[emotion] = os.listdir(dir_name) # TODO: precompute
        song_choices[emotion] = [dir_name + x for x in song_choices[emotion]]
    
    noise_files = []
    
    for label in labels:
        label = label_map[label] # convert from image annos to song
        choose_from = []
        for emotion in song_choices.keys():
            if label != emotion:
                choose_from.extend(song_choices[emotion]) # TODO: precompute
        noise_files.extend(np.random.choice(choose_from, config['num_noise_samples'], replace=False))
    
    return load_from_list(noise_files, config)

# Creates list of images to use as train/test/val.
image_lists = create_image_lists(config['image_dir'], config['testing_percentage'],
                                    config['validation_percentage'])

###################################
###    Load Pretrained Models   ###
###################################

# Set up the pre-trained image model.
module_spec = hub.load_module_spec(config['tfhub_module'])
graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
  create_module_graph(module_spec))

# Set up the pre-trained audio model.
param_G = np.load(config['param_g_dir'], encoding='latin1').item() \
        if config['phase'] in ['finetune', 'extract', 'train'] \
        else None
        
###################################
###       Graph Definition      ###
###################################

# Add the new embedding that we'll be training.
with graph.as_default():
    bottleneck_input, ground_truth_input, img_embedding = add_embedding_retrain_ops(config['hidden_dim'], bottleneck_tensor, False)
    
    # set up sound graph
    model = Model(config=config, param_G=param_G)
    song_emb = model.fetch_scene_embedding(config['hidden_dim'])
    true_song_emb = tf.slice(song_emb, [0, 0, 0, 0], [config['batch_size'], -1, -1, -1])
    noise_song_emb = tf.slice(song_emb, [config['batch_size'], 0, 0, 0], [-1, -1, -1, -1])
    true_song_emb = tf.reshape(true_song_emb, [config['batch_size'], -1])
    noise_song_emb = tf.reshape(noise_song_emb, [config['batch_size'], config['num_noise_samples'], -1])
    
    # construct loss
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(config['learning_rate'])
        nce_loss = NCE_sentiment_loss(img_embedding, true_song_emb, noise_song_emb)
        train_step = optimizer.minimize(nce_loss)
        
    # add evaluation step to check accuracy of model
    label_logits = tf.einsum('ij,ij->i', img_embedding, true_song_emb) # N
    noise_logits = tf.einsum('ij,igj->ig', img_embedding, noise_song_emb) # N x Noi
    preds = tf.concat([tf.expand_dims(label_logits, axis=1), noise_logits], axis=1)
    evaluation_step = tf.count_nonzero(tf.argmax(preds, axis=1)) # at least ensure true label is successfully selected from noise

###################################
### Session Creation + Training ###
###################################

with tf.Session(graph=graph) as sess:
    # Initialize all weights: for the module to their pretrained values,
    # and for the newly added retraining layer to random initial values.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

# TO DO: add distortion later; double check cache_bottlenecks checks if bottleneck already created
#     if do_distort_images:
#       # We will be applying distortions, so setup the operations we'll need.
#       (distorted_jpeg_data_tensor,
#        distorted_image_tensor) = add_input_distortions(
#            FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
#            FLAGS.random_brightness, module_spec)
    if True:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      cache_bottlenecks(sess, image_lists, config['image_dir'],
                        config['bottleneck_dir'], jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor,
                        bottleneck_tensor, config['tfhub_module'])

    

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(config['summaries_dir'] + '/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        config['summaries_dir'] + '/validation')

    # Create a train saver that is used to restore values into an eval graph
    # when exporting models.
    train_saver = tf.train.Saver()

    # Run the training for as many cycles as requested on the command line.
    for i in range(config['num_training_steps']):
      # Get a batch of input bottleneck values, either calculated fresh every
      # time with distortions applied, or from the cache stored on disk.
# TODO: change here if do distort images
#       if do_distort_images:
#         (train_bottlenecks,
#          train_ground_truth) = get_random_distorted_bottlenecks(
#              sess, image_lists, FLAGS.train_batch_size, 'training',
#              FLAGS.image_dir, distorted_jpeg_data_tensor,
#              distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
      if True:
        (train_bottlenecks,
         train_ground_truth, train_ground_truth_filenames) = get_random_cached_bottlenecks(
             sess, image_lists, config['train_batch_size'], 'training',
             config['bottleneck_dir'], config['image_dir'], jpeg_data_tensor,
             decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
             config['tfhub_module'])
        
      # use random cached bottleneck GT to sample songs
      # TODO: write sample song etc, read in from librosa
      songs = []
      labels = []
      for label in train_ground_truth_filenames:
          labels.append(label.split('/')[2])
      songs.extend(sample_song(labels))
      songs.extend(sample_noise_songs(labels, config['num_noise_samples']))
        
      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.
      model_layers = sess.run(
          [model.layers[22]],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth,
                     model.label_sound_placeholder: songs
                    })
      train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth,
                     model.label_sound_placeholder: songs
                    })
      train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == config['num_training_steps'])
      if (i % config['eval_step_interval']) == 0 or is_last_step:
        train_incorrect, nce_loss_value = sess.run(
            [evaluation_step, nce_loss],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth,
                       model.label_sound_placeholder: songs
                      })
        tf.logging.info('%s: Step %d: Percent Recovered from Noise = %.1f%%' %
                        (datetime.now(), i, 1 - (float(train_incorrect) / config['batch_size'])))
        tf.logging.info('%d incorrect from batch size %d' %
                        (train_incorrect, config['batch_size']))
        tf.logging.info('%s: Step %d: NCE loss = %f' %
                        (datetime.now(), i, nce_loss_value))
        
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, config['batch_size'], 'validation',
                config['bottleneck_dir'], config['image_dir'], jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                config['tfhub_module']))
        
        # find validation songs
        val_songs = []
        val_labels = []
        for label in train_ground_truth_filenames:
          val_labels.append(label.split('/')[2])
        val_songs.extend(sample_song(val_labels))
        val_songs.extend(sample_noise_songs(val_labels, config['num_noise_samples']))
        
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        val_incorrect = sess.run(
            [evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth,
                       model.label_sound_placeholder: val_songs
                      })
        # validation_writer.add_summary(validation_summary, i)
        tf.logging.info('%s: Step %d: Validation accuracy = %f%% (N=%d)' %
                        (datetime.now(), i, 1 - (val_incorrect[0] / config['batch_size']),
                         len(validation_bottlenecks)))
        
        # TODO: need to store model somewhere

#       # Store intermediate results
#       intermediate_frequency = FLAGS.intermediate_store_frequency

#       if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
#           and i > 0):
#         # If we want to do an intermediate save, save a checkpoint of the train
#         # graph, to restore into the eval graph.
#         train_saver.save(sess, CHECKPOINT_NAME)
#         intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
#                                   'intermediate_' + str(i) + '.pb')
#         tf.logging.info('Save intermediate result to : ' +
#                         intermediate_file_name)
#         save_graph_to_file(graph, intermediate_file_name, module_spec,
#                            class_count)

#     # After training is complete, force one last save of the train checkpoint.
#     train_saver.save(sess, CHECKPOINT_NAME)

#     # We've completed all our training, so run a final test evaluation on
#     # some new images we haven't used before.
#     run_final_eval(sess, module_spec, class_count, image_lists,
#                    jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
#                    bottleneck_tensor)

#     # Write out the trained graph and labels with the weights stored as
#     # constants.
#     tf.logging.info('Save final result to : ' + FLAGS.output_graph)
#     if wants_quantization:
#       tf.logging.info('The model is instrumented for quantization with TF-Lite')
#     save_graph_to_file(graph, FLAGS.output_graph, module_spec, class_count)
#     with tf.gfile.FastGFile(FLAGS.output_labels, 'w') as f:
#       f.write('\n'.join(image_lists.keys()) + '\n')

#     if FLAGS.saved_model_dir:
#       export_model(module_spec, class_count, FLAGS.saved_model_dir)
