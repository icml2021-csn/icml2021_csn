import numpy as np
import tensorflow as tf

from data_parsing.mnist_data import get_digit_data, make_noisy
from models.proto_model import ProtoModel
from utils.gpu import set_gpu_config

set_gpu_config()

print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

latent_dim = 2
noise_level = 0.0
# Get the MNIST data. Do you want digit or fashion data?
x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_digit_data()
# x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_fashion_data()

x_train_noisy = make_noisy(x_train, noise_level=noise_level)
x_test_noisy = make_noisy(x_test, noise_level=noise_level)
parity_train_one_hot = np.zeros((y_train.shape[0], 2))
for i, y in enumerate(y_train):
    parity_train_one_hot[i][y % 2] = 1
parity_test = np.zeros((y_test.shape[0]))
parity_test_one_hot = np.zeros((y_train.shape[0], 2))
for i, y in enumerate(y_test):
    parity_test[i] = y % 2
    parity_test_one_hot[i][y % 2] = 1


# Create method for corrupting images. This noise will have to be purged.
def corrupt_img(images, labels, parity=0, prob=0.8):
    new_images = np.copy(images)
    did_corrupt = np.zeros(images.shape[0])
    for idx, label in enumerate(labels):
        if (label % 2 == parity and np.random.random() < prob):  # or (label % 2 != parity and np.random.random() < 1 - prob):
            corrupting_signal = np.zeros((1, 784))
            # corrupting_signal[0, label * 70: (label + 1) * 70] = 1  # FIXME: just a standard watermark would do
            corrupting_signal[0, 0: 70] = 1
            new_images[idx] = np.maximum(new_images[idx], new_images[idx] + corrupting_signal)
            did_corrupt[idx] = 1
    return new_images, did_corrupt


x_train_noisy, corrupted_train = corrupt_img(x_train_noisy, y_train, parity=0)
x_train = x_train_noisy
x_test_noisy, corrupted_test = corrupt_img(x_test_noisy, y_test, parity=1)
x_test = x_test_noisy
corrupted_train_one_hot = tf.keras.utils.to_categorical(corrupted_train, num_classes=2)
output_sizes = [10, 2, 2]
one_hot_output = [y_train_one_hot, parity_train_one_hot, corrupted_train_one_hot]
output = [y_test, parity_test, corrupted_test]
classification_weights = [0, 10, 0.01]  # Mess with these weights as desired.
proto_dist_weights = [0, 1, 1]  # How realistic are the prototypes
feature_dist_weights = [0, 1, 1]  # How close to prototypes are embeddings (cluster size)
disentangle_weights = [[0 for _ in range(len(output_sizes))] for _ in range(len(output_sizes))]
# disentangle_weights[0] = [0, -10, 0]
# disentangle_weights[0] = [0, -10, 100000]
disentangle_weights[1] = [0, 0, 1000]
# kl_losses = [0, 0, 0]
kl_losses = [0, 10, 10]
duplication_factors = [1, 1, 1]
all_labels = [class_labels, ['even', 'odd'], ['intact', 'corrupted']]

parity_proto_grid = None
digit_proto_grid = None
all_proto_grids = [digit_proto_grid, parity_proto_grid]

# Run a bunch of trials.
for model_id in range(0, 10):
    # Create, train, and eval the model
    proto_model = ProtoModel(output_sizes, decode_weight=0, duplication_factors=duplication_factors, input_size=784,
                             classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=latent_dim, proto_grids=all_proto_grids,
                             in_plane_clusters=True, use_shadow_basis=False)
    # proto_model.load_model('../saved_models/', 0)
    proto_model.train(x_train_noisy, x_train, one_hot_output, batch_size=64, epochs=10)
    # Eval the train and test sets/
    y_accs, s_diff, alignments, prob_updates, disparate_impact, demographic = proto_model.evaluate(x_test_noisy, x_test, output, one_hot_output)
    # mst = proto_model.get_mst(add_origin=True, plot=True, labels=all_labels)
    # proto_model.viz_projected_latent_space(x_test, y_test, class_labels, proto_indices=0)
    # proto_model.viz_projected_latent_space(x_test, parity_test, class_labels, proto_indices=1)
    # proto_model.viz_projected_latent_space(x_test, corrupted_test, class_labels, proto_indices=2)
    proto_model.viz_latent_space(x_train, y_train, class_labels)
    proto_model.viz_latent_space(x_test, y_test, class_labels)
    # mean_diffs, mean_sames = proto_model.eval_proto_diffs_parity(concept_idx=1)  # Use parity as ground truth.
    # write_to_file([y_accs, alignments, mean_diffs, mean_sames, prob_updates[0], prob_updates[1]], filename='../saved_models/mnist_protos_' + str(model_id) + '.csv')
    tf.keras.backend.clear_session()
