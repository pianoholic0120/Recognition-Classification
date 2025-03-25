import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

def get_tiny_images(img_paths: str):
    tiny_img_feats = []
    for path in tqdm(img_paths):
        with Image.open(path) as img:
            img = img.resize((16, 16), Image.Resampling.BICUBIC)
            img = np.array(img, dtype=np.float32).flatten()
            img -= np.mean(img) # zero-mean
            norm = np.linalg.norm(img)
            if norm > 1e-6:
                img /= norm
            tiny_img_feats.append(img)

    tiny_img_feats = np.array(tiny_img_feats) # convert back to numpy array
    return tiny_img_feats

def build_vocabulary(img_paths: list, vocab_size: int = 400):
    sift_per_img = 200  # how many descripters to sample from each image
    all_descs = []

    for path in tqdm(img_paths):
        with Image.open(path) as img:
            img = np.array(img, dtype=np.float32)
            if len(img.shape) == 3:
                img = img[:, :, 0]
            _, descs = dsift(img, step=[5, 5], fast=True)
            if descs.shape[0] > sift_per_img:
                idx = np.random.choice(descs.shape[0], sift_per_img, replace=False)
                descs = descs[idx, :]
            all_descs.append(descs)

    all_descs = np.vstack(all_descs).astype(np.float32)
    vocab = kmeans(all_descs, num_centers=vocab_size)
    return vocab

def get_bags_of_sifts(img_paths: list,vocab: np.array):
    vocab_size = vocab.shape[0]
    img_feats = []

    for path in tqdm(img_paths):
        with Image.open(path) as img:
            img = np.array(img, dtype=np.float32)
            if len(img.shape) == 3:
                img = img[:, :, 0]
            _, descs = dsift(img, step=[5, 5], fast=True)
            descs = descs.astype(np.float32)

            dist = cdist(descs, vocab, metric='euclidean') # shape: (num_desc, vocab_size)
            labels = np.argmin(dist, axis=1) # the closest cluster center for each descriptor
            hist, _ = np.histogram(labels, bins=np.arange(vocab_size+1))
            # L1 normalization
            if hist.sum() > 0:
                hist = hist / hist.sum()
            img_feats.append(hist)

    img_feats = np.array(img_feats)
    return img_feats

from collections import Counter

def nearest_neighbor_classify(train_img_feats: np.array, train_labels: list, test_img_feats: list):
    dists = cdist(test_img_feats, train_img_feats, metric='cityblock')
    test_predicts = []

    for i in range(dists.shape[0]):
        nearest_idx = np.argsort(dists[i])[:5] # 5 nearest neighbors
        nearest_labels = [train_labels[idx] for idx in nearest_idx]
        vote_result = Counter(nearest_labels).most_common(1)[0][0]
        test_predicts.append(vote_result)

    return test_predicts
