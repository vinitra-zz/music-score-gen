"""
Implementation of consensus nearest neighbors for
image captioning described in https://arxiv.org/pdf/1505.01809.pdf.

Takes in fc7 vectors (output of CNN final hidden layer) and returns
batch of cpations.
"""

from sklearn.neighbors import NearestNeighbors
from numpy import argmax
import ngram, pickle, glob, json
import tensorflow as tf
import numpy as np

#############################################
#####  Data Utils Too Specific to COCO  #####
#############################################

def img_name_2_img_id(img_name):
    return int(img_name.split('.')[0].split('_')[3])

#############################################
########      Captioning Tools       ########
#############################################

class ConsensusNearestNeighbors():

    def process_training_set(self, pickle_file, train_dir, model, sess):
        """
        Preprocesses training set captions for more efficient NN.
        Pickles into pickle_file.

        Args:
          pickle_file: filename to save model to
          train_dir: directory with training images
        """
        self.k=90
        self.neigh = NearestNeighbors(n_neighbors=self.k, metric='cosine')

        # gather all image representations into list
        img_features, idx_map = [], {}
        for i, file_name in enumerate(glob.glob(train_dir + '*')):
            with tf.gfile.GFile(file_name, 'rb') as f:
                image = f.read()
            img_features.append(np.reshape(model.feed_image(sess, image), -1))
            idx_map[i] = file_name

        # create dictionary from indices to filenames
        self.idx_map = idx_map

        # fit model
        self.neigh.fit(img_features)

        # save model
        pickle.dump((self.neigh, self.idx_map), open(pickle_file, 'wb'))

    def load_model(self, pickle_file):
        self.neigh, self.idx_map = pickle.load(open(pickle_file, 'rb'))

    def consensus_caption(self, captions):
        """
        From list of captions returns the best 'consensus'
        caption: caption with largest average n-gram overlap
        with other captions in list.

        input: list of captions
        output: the consensus caption
        """
        caption_scores = [0] * len(captions)

        for i in range(len(captions)):
            for j in range(i+1, len(captions)):
                pairwise_scores = 0
                for n in [2, 3, 4]:
                   pairwise_scores += ngram.NGram.compare(captions[i], captions[j], N=n)
                pairwise_scores = pairwise_scores / float(n)
                caption_scores[i] = pairwise_scores
                caption_scores[j] = pairwise_scores 
        return captions[argmax(caption_scores)]


    def generate_captions(self, img_features, caption_file):
        """
        Generates captions for a batch of images.
        input: array of image features to match, shape
            (n_query, n_features)
        """
        predicted_captions = []
        query_img_indices = self.neigh.kneighbors(img_features, return_distance=False)

        # load caption data
        dataset = json.load(open(caption_file, 'r'))['annotations']
        ids2caps = {ann['image_id'] : ann['caption'] for ann in dataset}
        for img_indices in query_img_indices:
            # look up image names
            img_names = [self.idx_map[img_idx] for img_idx in img_indices]
            # find corresponding captions
            captions = {name: ids2caps[img_name_2_img_id(name)] for name in img_names}
            predicted_captions.append(self.consensus_caption(captions))
        return predicted_captions

