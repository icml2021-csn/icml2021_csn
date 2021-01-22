import numpy as np
import tensorflow as tf
from data_parsing.compas_data import get_compas_data
from models.proto_model import ProtoModel
import tensorflow.keras as keras

print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

disentangle = True

train_data, train_labels, train_protected, test_data, test_labels, test_protected = get_compas_data()
input_size = train_data.shape[1]

# Create one-hot encodings of data
train_labels_one_hot = keras.utils.to_categorical(train_labels, num_classes=2)
train_protected_one_hot = keras.utils.to_categorical(train_protected)
test_labels_one_hot = keras.utils.to_categorical(test_labels, num_classes=2)
test_protected_one_hot = keras.utils.to_categorical(test_protected)

num_protected_classes = train_protected_one_hot.shape[1]
if disentangle:
    output_sizes = [2, num_protected_classes]

    train_outputs_one_hot = [train_labels_one_hot, train_protected_one_hot]
    test_outputs = [test_labels, test_protected]
    test_outputs_one_hot = [test_labels_one_hot, test_protected_one_hot]
else:
    output_sizes = [2]  # Binary choice
    train_outputs_one_hot = [train_labels_one_hot]
    test_outputs = [test_labels]
    test_outputs_one_hot = [test_labels_one_hot]


mean_train_labels = np.mean(train_labels)
print("Mean test rate", mean_train_labels)
mean_test_rate = np.mean(test_labels)
print("Mean test rate", mean_test_rate)

classification_weight = [1] if not disentangle else [1, 1]
proto_dist_weights = [1] if not disentangle else [.1, .1]
feature_dist_weights = [1] if not disentangle else [.1, .1]
disentangle_weights = [[0, 1000], [0, 0]]
kl_losses = [10] if not disentangle else [20, 10]
proto_model = ProtoModel(output_sizes, input_size=input_size, decode_weight=0,
                         classification_weights=classification_weight, proto_dist_weights=proto_dist_weights,
                         feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                         kl_losses=kl_losses)

proto_model.train(train_data, train_data, train_outputs_one_hot, batch_size=64, epochs=20)
proto_model.evaluate(test_data, test_data, test_outputs, test_outputs_one_hot, plot=False)
proto_model.viz_latent_space(test_data, test_labels, [i for i in range(2)])
proto_model.viz_latent_space(test_data, test_protected, [i for i in range(5)], proto_indices=1)
