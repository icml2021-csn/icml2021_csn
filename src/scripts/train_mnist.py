import numpy as np
import tensorflow as tf

from data_parsing.mnist_data import get_digit_data, make_noisy, get_parity_tree
from models.proto_model import ProtoModel
from utils.eval import trees_match, graph_edit_dist
from utils.gpu import set_gpu_config
from utils.to_files import write_to_file

set_gpu_config()

print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()
# Set seed before getting the data.
np.random.seed(0)
tf.random.set_seed(0)

latent_dim = 32
noise_level = 0.0  # If you want to play with this, can blur images, but don't need to.
use_digit_and_parity = True
digit_only = False
parity_only = False
# Get the MNIST data. Do you want digit or fashion data?
x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_digit_data()
# x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_fashion_data()

x_train_noisy = make_noisy(x_train, noise_level=noise_level)
x_test_noisy = make_noisy(x_test, noise_level=noise_level)
parity_train_one_hot = np.zeros((y_train.shape[0], 2))
for i, y in enumerate(y_train):
    parity_train_one_hot[i][y % 2] = 1
parity_test = np.zeros((y_test.shape[0]))
parity_test_one_hot = np.zeros((y_test.shape[0], 2))
for i, y in enumerate(y_test):
    parity_test[i] = y % 2
    parity_test_one_hot[i][y % 2] = 1

if use_digit_and_parity:
    output_sizes = [10, 2]
    one_hot_output = [y_train_one_hot, parity_train_one_hot]
    output = [y_test, parity_test]
elif digit_only:
    output_sizes = [10]
    one_hot_output = [y_train_one_hot]
    output = [y_test]
elif parity_only:
    output_sizes = [2]
    one_hot_output = [parity_train_one_hot]
    output = [parity_test]
# Ground truth tree
ground_truth_tree = get_parity_tree()

# classification weights of [0.8, 10] seems good for showing worsened performance when disentanglement is on.
classification_weights = [10] if not use_digit_and_parity else [10, 10]  # Mess with these weights as desired.
proto_dist_weights = [1] if not use_digit_and_parity else [1, 1]  # How realistic are the prototypes
feature_dist_weights = [1] if not use_digit_and_parity else [1, 1]  # How close to prototypes are embeddings (cluster size)
disentangle_weights = [[0 for _ in range(2)] for _ in range(2)]
disentangle_weights[0] = [0, -10]
# kl losses of [10, 10] is good for bad performance with disentanglement on.
kl_losses = [1] if not use_digit_and_parity else [0, 0]
duplication_factors = [1] if not use_digit_and_parity else [1, 1]

# If you want to create grids for the prototypes
# parity_proto_grid = np.zeros((2, latent_dim))
# for i in range(2):
#     parity_proto_grid[i, 0] = i % 2
#     parity_proto_grid[i, 1] = 5
parity_proto_grid = None
# Each in a separate direction
# digit_proto_grid = np.eye(10, latent_dim, k=0)
# Create two parallel lines, all in a single plane.
# digit_proto_grid = np.zeros((10, latent_dim))
# for i in range(10):
#     x = 0 if i % 2 == 0 else 1
#     digit_proto_grid[i, 0] = x
#     digit_proto_grid[i, 1] = i
digit_proto_grid = None
all_proto_grids = [digit_proto_grid, parity_proto_grid]

# Run a bunch of trials.
for model_id in range(0, 10):
    np.random.seed(model_id)
    tf.random.set_seed(model_id)
    # Create, train, and eval the model
    proto_model = ProtoModel(output_sizes, duplication_factors=duplication_factors, input_size=784,
                             classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=latent_dim, proto_grids=all_proto_grids,
                             in_plane_clusters=True, use_shadow_basis=False, align_fn=tf.reduce_mean, network_type='dense_mnist')
    # proto_model.load_model('../saved_models/', 0)
    proto_model.train(x_train_noisy, x_train, one_hot_output, batch_size=128, epochs=00)
    y_accs, s_diff, alignments, prob_updates, disparate_impact, demographic, average_cost = proto_model.evaluate(x_test_noisy, x_test, output, one_hot_output, gold_tree=(ground_truth_tree, class_labels))
    mst = proto_model.get_mst(add_origin=True, plot=False, labels=[class_labels, ['even', 'odd']])
    tree_matches = trees_match(mst, ground_truth_tree)
    print("Tree matches", tree_matches)
    # Super fast if the trees are actually close together.
    edit_dist = graph_edit_dist(ground_truth_tree, mst)
    # edit_dist = 0
    print("Edit distance", edit_dist)
    # proto_model.viz_projected_latent_space(x_test, y_test, class_labels, proto_indices=0)
    # proto_model.viz_projected_latent_space(x_test, parity_test, class_labels, proto_indices=1)
    # proto_model.viz_latent_space(x_test, y_test, class_labels)
    mean_diffs, mean_sames = proto_model.eval_proto_diffs_parity(concept_idx=1)  # Use parity as ground truth.
    # write_to_file([y_accs, alignments, mean_diffs, mean_sames, prob_updates[0], prob_updates[1]], filename='../saved_models/mnist_protos_' + str(model_id) + '.csv')
    # write_to_file([y_accs, alignments, mean_diffs, mean_sames, [1] if tree_matches else [0], [edit_dist], [average_cost]], filename='../saved_models/mnist_digit_cost_' + str(model_id) + '.csv')
    write_to_file([y_accs, alignments, mean_diffs, mean_sames, [1] if tree_matches else [0], [edit_dist], [average_cost]], filename='../saved_models/random_mnist_cost_' + str(model_id) + '.csv')
    tf.keras.backend.clear_session()
