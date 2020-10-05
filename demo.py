import os
import pickle
import shutil
import time
from collections import OrderedDict
from datetime import datetime
from itertools import product

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app, flags
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Dense,
                                     GlobalAveragePooling2D, Input, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

import applications
import image_augmentation
from datasets import load_accumulated_info_of_dataset
from evaluation.metrics import compute_CMC_mAP
from evaluation.post_processing.re_ranking_ranklist import re_ranking
from metric_learning.triplet_hermans import batch_hard, cdist
from utils.model_utils import replicate_model, specify_regularizers
from utils.vis_utils import visualize_model
from visualizer import Visualizer

# Specify the backend of matplotlib
matplotlib.use("Agg")

flags.DEFINE_string("root_folder_path", "", "Folder path of the dataset.")
flags.DEFINE_string("dataset_name", "Market1501", "Name of the dataset.")
# ["Market1501", "DukeMTMC_reID", "MSMT17"]
flags.DEFINE_string("backbone_model_name", "ResNet50",
                    "Name of the backbone model.")
# ["ResNet50", "ResNet101", "ResNet152",
# "ResNet50V2", "ResNet101V2", "ResNet152V2",
# "ResNeXt50", "ResNeXt101"]
flags.DEFINE_integer("freeze_backbone_for_N_epochs", 20,
                     "Freeze layers in the backbone model for N epochs.")
flags.DEFINE_integer("image_width", 128, "Width of the images.")
flags.DEFINE_integer("image_height", 384, "Height of the images.")
flags.DEFINE_integer("region_num", 2,
                     "Number of regions in the regional branch.")
flags.DEFINE_float("kernel_regularization_factor", 0.005,
                   "Regularization factor of kernel.")
flags.DEFINE_float("bias_regularization_factor", 0.005,
                   "Regularization factor of bias.")
flags.DEFINE_float("gamma_regularization_factor", 0.005,
                   "Regularization factor of gamma.")
flags.DEFINE_float("beta_regularization_factor", 0.005,
                   "Regularization factor of beta.")
flags.DEFINE_bool("use_adaptive_l1_l2_regularizer", True,
                  "Use the adaptive L1L2 regularizer.")
flags.DEFINE_float("min_value_in_clipping", 0.0,
                   "Minimum value when using the clipping function.")
flags.DEFINE_float("max_value_in_clipping", 1.0,
                   "Maximum value when using the clipping function.")
flags.DEFINE_float("validation_size", 0.0,
                   "Proportion or absolute number of validation samples.")
flags.DEFINE_float("testing_size", 1.0,
                   "Proportion or absolute number of testing groups.")
flags.DEFINE_integer(
    "evaluate_validation_every_N_epochs", 1,
    "Evaluate the performance on validation samples every N epochs.")
flags.DEFINE_integer(
    "evaluate_testing_every_N_epochs", 10,
    "Evaluate the performance on testing samples every N epochs.")
flags.DEFINE_integer("identity_num_per_batch", 16,
                     "Number of identities in one batch.")
flags.DEFINE_integer("image_num_per_identity", 4,
                     "Number of images of one identity.")
flags.DEFINE_string("learning_rate_mode", "default",
                    "Mode of the learning rate scheduler.")
# ["constant", "linear", "cosine", "warmup", "default"]
flags.DEFINE_float("learning_rate_start", 2e-4, "Starting learning rate.")
flags.DEFINE_float("learning_rate_end", 2e-4, "Ending learning rate.")
flags.DEFINE_float("learning_rate_base", 2e-4, "Base learning rate.")
flags.DEFINE_integer("learning_rate_warmup_epochs", 10,
                     "Number of epochs to warmup the learning rate.")
flags.DEFINE_integer("learning_rate_steady_epochs", 30,
                     "Number of epochs to keep the learning rate steady.")
flags.DEFINE_float("learning_rate_drop_factor", 10,
                   "Factor to decrease the learning rate.")
flags.DEFINE_float("learning_rate_lower_bound", 2e-6,
                   "Lower bound of the learning rate.")
flags.DEFINE_integer("steps_per_epoch", 200, "Number of steps per epoch.")
flags.DEFINE_integer("epoch_num", 200, "Number of epochs.")
flags.DEFINE_integer("workers", 5,
                     "Number of processes to spin up for data generator.")
flags.DEFINE_string("image_augmentor_name", "RandomErasingImageAugmentor",
                    "Name of image augmentor.")
# ["BaseImageAugmentor", "RandomErasingImageAugmentor"]
flags.DEFINE_bool("use_data_augmentation_in_training", True,
                  "Use data augmentation in training.")
flags.DEFINE_bool("use_data_augmentation_in_evaluation", False,
                  "Use data augmentation in evaluation.")
flags.DEFINE_integer("augmentation_num", 1,
                     "Number of augmented samples to use in evaluation.")
flags.DEFINE_bool("use_horizontal_flipping_in_evaluation", True,
                  "Use horizontal flipping in evaluation.")
flags.DEFINE_bool("use_label_smoothing_in_training", True,
                  "Use label smoothing in training.")
flags.DEFINE_bool("use_identity_balancing_in_training", False,
                  "Use identity balancing in training.")
flags.DEFINE_bool("use_re_ranking", False, "Use the re-ranking method.")
flags.DEFINE_bool("evaluation_only", False, "Only perform evaluation.")
flags.DEFINE_bool("save_data_to_disk", False,
                  "Save image features, identity ID and camera ID to disk.")
flags.DEFINE_string("pretrained_model_file_path", "",
                    "File path of the pretrained model.")
flags.DEFINE_string(
    "output_folder_path",
    os.path.abspath(
        os.path.join(__file__, "../output_{}".format(
            datetime.now().strftime("%Y_%m_%d")))),
    "Path to directory to output files.")
FLAGS = flags.FLAGS


def init_model(backbone_model_name,
               freeze_backbone_for_N_epochs,
               input_shape,
               region_num,
               attribute_name_to_label_encoder_dict,
               kernel_regularization_factor,
               bias_regularization_factor,
               gamma_regularization_factor,
               beta_regularization_factor,
               use_adaptive_l1_l2_regularizer,
               min_value_in_clipping,
               max_value_in_clipping,
               share_last_block=False):

    def _add_objective_module(input_tensor):
        # Add GlobalAveragePooling2D
        if len(K.int_shape(input_tensor)) == 4:
            global_average_pooling_tensor = GlobalAveragePooling2D()(
                input_tensor)
        else:
            global_average_pooling_tensor = input_tensor
        if min_value_in_clipping is not None and max_value_in_clipping is not None:
            global_average_pooling_tensor = Lambda(lambda x: K.clip(
                x,
                min_value=min_value_in_clipping,
                max_value=max_value_in_clipping))(global_average_pooling_tensor)

        # https://arxiv.org/abs/1801.07698v1 Section 3.2.2 Output setting
        # https://arxiv.org/abs/1807.11042
        classification_input_tensor = global_average_pooling_tensor
        classification_embedding_tensor = BatchNormalization(
            scale=True, epsilon=2e-5)(classification_input_tensor)

        # Add categorical crossentropy loss
        assert len(attribute_name_to_label_encoder_dict) == 1
        # label_encoder = attribute_name_to_label_encoder_dict["identity_ID"]
        # class_num = len(label_encoder.classes_)
        # TODO: hardcoded for Market1501 model
        class_num = 751
        classification_output_tensor = Dense(
            units=class_num,
            use_bias=False,
            kernel_initializer=RandomNormal(
                mean=0.0, stddev=0.001))(classification_embedding_tensor)
        classification_output_tensor = Activation("softmax")(
            classification_output_tensor)

        # Add miscellaneous loss
        miscellaneous_input_tensor = global_average_pooling_tensor
        miscellaneous_embedding_tensor = miscellaneous_input_tensor
        miscellaneous_output_tensor = miscellaneous_input_tensor

        return classification_output_tensor, classification_embedding_tensor, miscellaneous_output_tensor, miscellaneous_embedding_tensor

    def _apply_concatenation(tensor_list):
        if len(tensor_list) == 1:
            return tensor_list[0]
        else:
            return Concatenate()(tensor_list)

    def _triplet_hermans_loss(y_true,
                              y_pred,
                              metric="euclidean",
                              margin="soft"):
        # Create the loss in two steps:
        # 1. Compute all pairwise distances according to the specified metric.
        # 2. For each anchor along the first dimension, compute its loss.
        dists = cdist(y_pred, y_pred, metric=metric)
        loss = batch_hard(dists=dists,
                          pids=tf.argmax(y_true, axis=-1),
                          margin=margin)
        return loss

    # Initiation
    classification_output_tensor_list = []
    classification_embedding_tensor_list = []
    miscellaneous_output_tensor_list = []
    miscellaneous_embedding_tensor_list = []

    # Initiate the early blocks
    model_instantiation = getattr(applications, backbone_model_name, None)
    assert model_instantiation is not None, "Backbone {} is not supported.".format(
        backbone_model_name)
    submodel_list, preprocess_input = model_instantiation(
        input_shape=input_shape)
    vanilla_input_tensor = Input(shape=K.int_shape(submodel_list[0].input)[1:])
    intermediate_output_tensor = vanilla_input_tensor
    for submodel in submodel_list[:-1]:
        if freeze_backbone_for_N_epochs > 0:
            submodel.trainable = False
        intermediate_output_tensor = submodel(intermediate_output_tensor)

    # Initiate the last blocks
    last_block = submodel_list[-1]
    last_block_for_global_branch_model = replicate_model(
        last_block, name="last_block_for_global_branch")
    if freeze_backbone_for_N_epochs > 0:
        last_block_for_global_branch_model.trainable = False
    if share_last_block:
        last_block_for_regional_branch_model = last_block_for_global_branch_model
    else:
        last_block_for_regional_branch_model = replicate_model(
            last_block, name="last_block_for_regional_branch")
        if freeze_backbone_for_N_epochs > 0:
            last_block_for_regional_branch_model.trainable = False

    # Add the global branch
    classification_output_tensor, classification_embedding_tensor, miscellaneous_output_tensor, miscellaneous_embedding_tensor = _add_objective_module(
        last_block_for_global_branch_model(intermediate_output_tensor))
    classification_output_tensor_list.append(classification_output_tensor)
    classification_embedding_tensor_list.append(classification_embedding_tensor)
    miscellaneous_output_tensor_list.append(miscellaneous_output_tensor)
    miscellaneous_embedding_tensor_list.append(miscellaneous_embedding_tensor)

    # Add the regional branch
    if region_num > 0:
        # Process each region
        regional_branch_output_tensor = last_block_for_regional_branch_model(
            intermediate_output_tensor)
        total_height = K.int_shape(regional_branch_output_tensor)[1]
        region_size = total_height // region_num
        for region_index in np.arange(region_num):
            # Get a slice of feature maps
            start_index = region_index * region_size
            end_index = (region_index + 1) * region_size
            if region_index == region_num - 1:
                end_index = total_height
            sliced_regional_branch_output_tensor = Lambda(
                lambda x, start_index=start_index, end_index=end_index:
                x[:, start_index:end_index])(regional_branch_output_tensor)

            # Downsampling
            sliced_regional_branch_output_tensor = Conv2D(
                filters=K.int_shape(sliced_regional_branch_output_tensor)[-1] //
                region_num,
                kernel_size=3,
                padding="same")(sliced_regional_branch_output_tensor)
            sliced_regional_branch_output_tensor = Activation("relu")(
                sliced_regional_branch_output_tensor)

            # Add the regional branch
            classification_output_tensor, classification_embedding_tensor, miscellaneous_output_tensor, miscellaneous_embedding_tensor = _add_objective_module(
                sliced_regional_branch_output_tensor)
            classification_output_tensor_list.append(
                classification_output_tensor)
            classification_embedding_tensor_list.append(
                classification_embedding_tensor)
            miscellaneous_output_tensor_list.append(miscellaneous_output_tensor)
            miscellaneous_embedding_tensor_list.append(
                miscellaneous_embedding_tensor)

    # Define the merged model
    embedding_tensor_list = [
        _apply_concatenation(miscellaneous_embedding_tensor_list)
    ]
    embedding_size_list = [
        K.int_shape(embedding_tensor)[1]
        for embedding_tensor in embedding_tensor_list
    ]
    merged_embedding_tensor = _apply_concatenation(embedding_tensor_list)
    merged_model = Model(inputs=[vanilla_input_tensor],
                         outputs=classification_output_tensor_list +
                         miscellaneous_output_tensor_list +
                         [merged_embedding_tensor])
    merged_model = specify_regularizers(merged_model,
                                        kernel_regularization_factor,
                                        bias_regularization_factor,
                                        gamma_regularization_factor,
                                        beta_regularization_factor,
                                        use_adaptive_l1_l2_regularizer)

    # Define the models for training/inference
    training_model = Model(inputs=[merged_model.input],
                           outputs=merged_model.output[:-1],
                           name="training_model")
    inference_model = Model(inputs=[merged_model.input],
                            outputs=[merged_model.output[-1]],
                            name="inference_model")
    inference_model.embedding_size_list = embedding_size_list

    # Compile the model
    classification_loss_function_list = [
        "categorical_crossentropy"
    ] * len(classification_output_tensor_list)
    triplet_hermans_loss_function = lambda y_true, y_pred: 1.0 * _triplet_hermans_loss(
        y_true, y_pred)
    miscellaneous_loss_function_list = [triplet_hermans_loss_function
                                       ] * len(miscellaneous_output_tensor_list)
    training_model.compile_kwargs = {
        "optimizer":
            Adam(),
        "loss":
            classification_loss_function_list + miscellaneous_loss_function_list
    }
    training_model.compile(**training_model.compile_kwargs)

    # Print the summary of the models
    # summarize_model(training_model)
    # summarize_model(inference_model)

    return training_model, inference_model, preprocess_input


def read_image_file(image_file_path, input_shape):
    # Read image file
    image_content = cv2.imread(image_file_path)

    # Resize the image
    image_content = cv2.resize(image_content, input_shape[:2][::-1])

    # Convert from BGR to RGB
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)

    return image_content


def apply_label_smoothing(y_true, epsilon=0.1):
    # https://arxiv.org/abs/1512.00567
    # https://github.com/keras-team/keras/pull/4723
    # https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/tools/loss.py#L6
    y_true = (1 - epsilon) * y_true + epsilon / y_true.shape[1]
    return y_true


class TrainDataSequence(Sequence):

    def __init__(self, accumulated_info_dataframe,
                 attribute_name_to_label_encoder_dict, preprocess_input,
                 input_shape, image_augmentor, use_data_augmentation,
                 use_identity_balancing, use_label_smoothing,
                 label_repetition_num, identity_num_per_batch,
                 image_num_per_identity, steps_per_epoch):
        super(TrainDataSequence, self).__init__()

        # Save as variables
        self.accumulated_info_dataframe, self.attribute_name_to_label_encoder_dict, self.preprocess_input, self.input_shape = accumulated_info_dataframe, attribute_name_to_label_encoder_dict, preprocess_input, input_shape
        self.image_augmentor, self.use_data_augmentation, self.use_identity_balancing = image_augmentor, use_data_augmentation, use_identity_balancing
        self.use_label_smoothing, self.label_repetition_num = use_label_smoothing, label_repetition_num
        self.identity_num_per_batch, self.image_num_per_identity, self.steps_per_epoch = identity_num_per_batch, image_num_per_identity, steps_per_epoch

        # Unpack image_file_path and identity_ID
        self.image_file_path_array, self.identity_ID_array = self.accumulated_info_dataframe[
            ["image_file_path", "identity_ID"]].values.transpose()
        self.image_file_path_to_record_index_dict = dict([
            (image_file_path, record_index)
            for record_index, image_file_path in enumerate(
                self.image_file_path_array)
        ])
        self.batch_size = identity_num_per_batch * image_num_per_identity
        self.image_num_per_epoch = self.batch_size * steps_per_epoch

        # Initiation
        self.image_file_path_list_generator = self._get_image_file_path_list_generator(
        )
        self.image_file_path_list = next(self.image_file_path_list_generator)

    def _get_image_file_path_list_generator(self):
        # Map identity ID to image file paths
        identity_ID_to_image_file_paths_dict = {}
        for image_file_path, identity_ID in zip(self.image_file_path_array,
                                                self.identity_ID_array):
            if identity_ID not in identity_ID_to_image_file_paths_dict:
                identity_ID_to_image_file_paths_dict[identity_ID] = []
            identity_ID_to_image_file_paths_dict[identity_ID].append(
                image_file_path)

        image_file_path_list = []
        while True:
            # Split image file paths into multiple sections
            identity_ID_to_image_file_paths_in_sections_dict = {}
            for identity_ID in identity_ID_to_image_file_paths_dict.keys():
                image_file_paths = np.array(
                    identity_ID_to_image_file_paths_dict[identity_ID])
                if len(image_file_paths) < self.image_num_per_identity:
                    continue
                np.random.shuffle(image_file_paths)
                section_num = int(
                    len(image_file_paths) / self.image_num_per_identity)
                image_file_paths = image_file_paths[:section_num *
                                                    self.image_num_per_identity]
                image_file_paths_in_sections = np.split(image_file_paths,
                                                        section_num)
                identity_ID_to_image_file_paths_in_sections_dict[
                    identity_ID] = image_file_paths_in_sections

            while len(identity_ID_to_image_file_paths_in_sections_dict) \
                    >= self.identity_num_per_batch:
                # Choose identity_num_per_batch identity_IDs
                identity_IDs = np.random.choice(
                    list(identity_ID_to_image_file_paths_in_sections_dict.keys(
                    )),
                    size=self.identity_num_per_batch,
                    replace=False)
                for identity_ID in identity_IDs:
                    # Get one section
                    image_file_paths_in_sections = identity_ID_to_image_file_paths_in_sections_dict[
                        identity_ID]
                    image_file_paths = image_file_paths_in_sections.pop(-1)
                    if self.use_identity_balancing or len(
                            image_file_paths_in_sections) == 0:
                        del identity_ID_to_image_file_paths_in_sections_dict[
                            identity_ID]

                    # Add the entries
                    image_file_path_list += image_file_paths.tolist()

                if len(image_file_path_list) == self.image_num_per_epoch:
                    yield image_file_path_list
                    image_file_path_list = []

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        image_content_list, attribute_name_to_one_hot_encoding_list_dict = [], OrderedDict(
            {})
        image_file_path_list = self.image_file_path_list[index *
                                                         self.batch_size:
                                                         (index + 1) *
                                                         self.batch_size]
        for image_file_path in image_file_path_list:
            # Read image
            image_content = read_image_file(image_file_path, self.input_shape)
            image_content_list.append(image_content)

            # Get current record from accumulated_info_dataframe
            record_index = self.image_file_path_to_record_index_dict[
                image_file_path]
            accumulated_info = self.accumulated_info_dataframe.iloc[
                record_index]
            assert image_file_path == accumulated_info["image_file_path"]
            for attribute_name, label_encoder in self.attribute_name_to_label_encoder_dict.items(
            ):
                # Get the one hot encoding vector
                attribute_value = accumulated_info[attribute_name]
                one_hot_encoding = np.zeros(len(label_encoder.classes_))
                one_hot_encoding[label_encoder.transform([attribute_value
                                                         ])[0]] = 1

                # Append one_hot_encoding
                if attribute_name not in attribute_name_to_one_hot_encoding_list_dict:
                    attribute_name_to_one_hot_encoding_list_dict[
                        attribute_name] = []
                attribute_name_to_one_hot_encoding_list_dict[
                    attribute_name].append(one_hot_encoding)
        assert len(image_content_list) == self.batch_size

        # Construct image_content_array
        image_content_array = np.array(image_content_list)
        if self.use_data_augmentation:
            # Apply data augmentation
            image_content_array = self.image_augmentor.apply_augmentation(
                image_content_array)
        # Apply preprocess_input function
        image_content_array = self.preprocess_input(image_content_array)

        # Construct one_hot_encoding_array_list
        one_hot_encoding_array_list = []
        for one_hot_encoding_list in attribute_name_to_one_hot_encoding_list_dict.values(
        ):
            one_hot_encoding_array = np.array(one_hot_encoding_list)
            if self.use_label_smoothing:
                # Apply label smoothing
                one_hot_encoding_array = apply_label_smoothing(
                    one_hot_encoding_array)
            one_hot_encoding_array_list.append(one_hot_encoding_array)

        # Hacky solution to specify one_hot_encoding_array_list
        assert list(attribute_name_to_one_hot_encoding_list_dict.keys()
                   )[0] == "identity_ID"
        one_hot_encoding_array_list = [one_hot_encoding_array_list[0]
                                      ] * self.label_repetition_num

        return image_content_array, one_hot_encoding_array_list

    def on_epoch_end(self):
        self.image_file_path_list = next(self.image_file_path_list_generator)


class TestDataSequence(Sequence):

    def __init__(self, accumulated_info_dataframe, preprocess_input,
                 input_shape, image_augmentor, use_data_augmentation,
                 batch_size):
        super(TestDataSequence, self).__init__()

        # Save as variables
        self.accumulated_info_dataframe, self.preprocess_input, self.input_shape = accumulated_info_dataframe, preprocess_input, input_shape
        self.image_augmentor, self.use_data_augmentation = image_augmentor, use_data_augmentation

        # Unpack image_file_path and identity_ID
        self.image_file_path_array = self.accumulated_info_dataframe[
            "image_file_path"].values
        self.batch_size = batch_size
        self.steps_per_epoch = int(
            np.ceil(len(self.image_file_path_array) / self.batch_size))

        # Initiation
        self.image_file_path_list = self.image_file_path_array.tolist()
        self.use_horizontal_flipping = False

    def enable_horizontal_flipping(self):
        self.use_horizontal_flipping = True

    def disable_horizontal_flipping(self):
        self.use_horizontal_flipping = False

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        image_content_list = []
        image_file_path_list = self.image_file_path_list[index *
                                                         self.batch_size:
                                                         (index + 1) *
                                                         self.batch_size]
        for image_file_path in image_file_path_list:
            # Read image
            image_content = read_image_file(image_file_path, self.input_shape)
            if self.use_horizontal_flipping:
                image_content = cv2.flip(image_content, 1)
            image_content_list.append(image_content)

        # Construct image_content_array
        image_content_array = np.array(image_content_list)
        if self.use_data_augmentation:
            # Apply data augmentation
            image_content_array = self.image_augmentor.apply_augmentation(
                image_content_array)
        # Apply preprocess_input function
        image_content_array = self.preprocess_input(image_content_array)

        return image_content_array


def main(_):
    root_folder_path, dataset_name = FLAGS.root_folder_path, FLAGS.dataset_name
    backbone_model_name, freeze_backbone_for_N_epochs = FLAGS.backbone_model_name, FLAGS.freeze_backbone_for_N_epochs
    image_height, image_width = FLAGS.image_height, FLAGS.image_width
    input_shape = (image_height, image_width, 3)
    region_num = FLAGS.region_num
    kernel_regularization_factor = FLAGS.kernel_regularization_factor
    bias_regularization_factor = FLAGS.bias_regularization_factor
    gamma_regularization_factor = FLAGS.gamma_regularization_factor
    beta_regularization_factor = FLAGS.beta_regularization_factor
    use_adaptive_l1_l2_regularizer = FLAGS.use_adaptive_l1_l2_regularizer
    min_value_in_clipping, max_value_in_clipping = FLAGS.min_value_in_clipping, FLAGS.max_value_in_clipping
    identity_num_per_batch, image_num_per_identity = FLAGS.identity_num_per_batch, FLAGS.image_num_per_identity
    batch_size = identity_num_per_batch * image_num_per_identity
    steps_per_epoch = FLAGS.steps_per_epoch
    workers = FLAGS.workers
    use_multiprocessing = workers > 1
    image_augmentor_name = FLAGS.image_augmentor_name
    use_data_augmentation_in_training = FLAGS.use_data_augmentation_in_training
    use_data_augmentation_in_evaluation = FLAGS.use_data_augmentation_in_evaluation
    augmentation_num = FLAGS.augmentation_num
    use_horizontal_flipping_in_evaluation = FLAGS.use_horizontal_flipping_in_evaluation
    use_label_smoothing_in_training = FLAGS.use_label_smoothing_in_training
    use_identity_balancing_in_training = FLAGS.use_identity_balancing_in_training
    use_re_ranking = FLAGS.use_re_ranking
    pretrained_model_file_path = FLAGS.pretrained_model_file_path

    output_folder_path = os.path.abspath(
        os.path.join(
            FLAGS.output_folder_path,
            "{}_{}x{}".format(dataset_name, input_shape[0], input_shape[1]),
            "{}_{}_{}".format(backbone_model_name, identity_num_per_batch,
                              image_num_per_identity)))
    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path)
    print("Recreating the output folder at {} ...".format(output_folder_path))

    print("Loading the annotations of the {} dataset ...".format(dataset_name))
    train_and_valid_accumulated_info_dataframe, test_query_accumulated_info_dataframe, \
        test_gallery_accumulated_info_dataframe, train_and_valid_attribute_name_to_label_encoder_dict = \
        load_accumulated_info_of_dataset(root_folder_path=root_folder_path, dataset_name=dataset_name)

    print("Initiating the model ...")
    training_model, inference_model, preprocess_input = init_model(
        backbone_model_name=backbone_model_name,
        freeze_backbone_for_N_epochs=freeze_backbone_for_N_epochs,
        input_shape=input_shape,
        region_num=region_num,
        attribute_name_to_label_encoder_dict=
        train_and_valid_attribute_name_to_label_encoder_dict,
        kernel_regularization_factor=kernel_regularization_factor,
        bias_regularization_factor=bias_regularization_factor,
        gamma_regularization_factor=gamma_regularization_factor,
        beta_regularization_factor=beta_regularization_factor,
        use_adaptive_l1_l2_regularizer=use_adaptive_l1_l2_regularizer,
        min_value_in_clipping=min_value_in_clipping,
        max_value_in_clipping=max_value_in_clipping)

    print("Initiating the image augmentor {} ...".format(image_augmentor_name))
    image_augmentor = getattr(image_augmentation,
                              image_augmentor_name)(image_height=image_height,
                                                    image_width=image_width)
    image_augmentor.compose_transforms()

    # Model loading
    train_accumulated_info_dataframe = train_and_valid_accumulated_info_dataframe
    train_generator = TrainDataSequence(
        accumulated_info_dataframe=train_accumulated_info_dataframe,
        attribute_name_to_label_encoder_dict=
        train_and_valid_attribute_name_to_label_encoder_dict,
        preprocess_input=preprocess_input,
        input_shape=input_shape,
        image_augmentor=image_augmentor,
        use_data_augmentation=use_data_augmentation_in_training,
        use_identity_balancing=use_identity_balancing_in_training,
        use_label_smoothing=use_label_smoothing_in_training,
        label_repetition_num=len(training_model.outputs),
        identity_num_per_batch=identity_num_per_batch,
        image_num_per_identity=image_num_per_identity,
        steps_per_epoch=steps_per_epoch)

    assert os.path.isfile(pretrained_model_file_path)
    print("Loading weights from {} ...".format(pretrained_model_file_path))
    # Hacky workaround for the issue with "load_weights"
    if use_adaptive_l1_l2_regularizer:
        _ = training_model.test_on_batch(train_generator[0])
    training_model.load_weights(pretrained_model_file_path)

    print("Freezing the whole model in the evaluation_only mode ...")
    training_model.trainable = False
    training_model.compile(**training_model.compile_kwargs)

    def extract_features(data_generator):
        print("Extracting features...")
        # Extract the accumulated_feature_array
        accumulated_feature_array = None
        for _ in np.arange(augmentation_num):
            data_generator.disable_horizontal_flipping()
            feature_array = inference_model.predict(
                x=data_generator,
                workers=workers,
                use_multiprocessing=use_multiprocessing)
            if use_horizontal_flipping_in_evaluation:
                data_generator.enable_horizontal_flipping()
                feature_array += inference_model.predict(
                    x=data_generator,
                    workers=workers,
                    use_multiprocessing=use_multiprocessing)
                feature_array /= 2
            if accumulated_feature_array is None:
                accumulated_feature_array = feature_array / augmentation_num
            else:
                accumulated_feature_array += feature_array / augmentation_num
        return accumulated_feature_array

    def compute_distance_matrix(query_image_features, gallery_image_features,
                                metric, use_re_ranking):
        print("Computing distance matrix...")
        # Compute the distance matrix
        query_gallery_distance = pairwise_distances(query_image_features,
                                                    gallery_image_features,
                                                    metric=metric)
        distance_matrix = query_gallery_distance

        # Use the re-ranking method
        if use_re_ranking:
            query_query_distance = pairwise_distances(query_image_features,
                                                      query_image_features,
                                                      metric=metric)
            gallery_gallery_distance = pairwise_distances(
                gallery_image_features, gallery_image_features, metric=metric)
            distance_matrix = re_ranking(query_gallery_distance,
                                         query_query_distance,
                                         gallery_gallery_distance)

        return distance_matrix

    query_generator = TestDataSequence(
        test_query_accumulated_info_dataframe, preprocess_input, input_shape,
        image_augmentor, use_data_augmentation_in_evaluation, batch_size)

    gallery_generator = TestDataSequence(
        test_gallery_accumulated_info_dataframe, preprocess_input, input_shape,
        image_augmentor, use_data_augmentation_in_evaluation, batch_size)

    query_features = extract_features(query_generator)
    gallery_features = extract_features(gallery_generator)
    print(query_features.shape)
    print(gallery_features.shape)

    distance_matrix = compute_distance_matrix(query_features, gallery_features,
                                              "cosine", use_re_ranking)
    print(distance_matrix.shape)

    # Compute the CMC and mAP scores
    query_identity_id_array, query_camera_id_array = \
        test_query_accumulated_info_dataframe[
            ["identity_ID", "camera_ID"]].values.transpose()
    gallery_identity_id_array, gallery_camera_id_array = \
        test_gallery_accumulated_info_dataframe[
            ["identity_ID", "camera_ID"]].values.transpose()
    print(query_identity_id_array[0], query_camera_id_array[0])
    cmc_score_array, map_score = compute_CMC_mAP(
        distmat=distance_matrix,
        q_pids=query_identity_id_array,
        g_pids=gallery_identity_id_array,
        q_camids=query_camera_id_array,
        g_camids=gallery_camera_id_array)
    print(cmc_score_array, map_score)

    # TODO: WIP
    g_dataset_pd = test_gallery_accumulated_info_dataframe[
        ['image_file_path', 'identity_ID', 'camera_ID']]
    g_dataset = [tuple(x) for x in g_dataset_pd.to_numpy()]
    q_dataset_pd = test_query_accumulated_info_dataframe[
        ['image_file_path', 'identity_ID', 'camera_ID']]
    q_dataset = [tuple(x) for x in q_dataset_pd.to_numpy()]
    visualizer = Visualizer(g_dataset, q_dataset)
    visualizer.run(distance_matrix, num=200)

    num_query, num_gallery = distance_matrix.shape
    for i in range(num_query):
        distances = [distance_matrix[i, j] for j in range(num_gallery)]
        min_distance = min(distances)

        if min_distance <= 0.17:
            gallery_idx = distances.index(min_distance)
            gallery_id = gallery_identity_id_array[gallery_idx]
            gallery_camera_id = gallery_camera_id_array[gallery_idx]
            query_id = query_identity_id_array[i]
            query_camera_id = query_camera_id_array[i]
            print("Identity {}.{} is closest ({:.4f}) to identity {}.{}.".format(
                query_camera_id, query_id, min_distance, gallery_camera_id,
                gallery_id))

    print("All done!")


if __name__ == "__main__":
    app.run(main)
