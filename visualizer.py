"""This module contains a Visualizer class to visualize re-identification query
images, gallery images and results."""

import os
import os.path as osp

import numpy as np
import cv2 as cv
from tqdm import tqdm

GRID_SPACING = 10
Q_SPACING = 90
BW = 5  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class Visualizer:

    def __init__(self, g_dataset, q_dataset,
                 out_dir='output/', k=10):
        self.g_dataset = g_dataset
        self.q_dataset = q_dataset
        self.out_dir = out_dir
        self.k = k

        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

    @staticmethod
    def save_image(image, image_path):
        # image = image.numpy().transpose((1, 2, 0)) * 255
        # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(image_path, image)

    @staticmethod
    def read_image(image_path):
        return cv.imread(image_path)

    @staticmethod
    def save_query_images(images_dir, image_tuple):
        image_path, p_id, cam_id = image_tuple
        image = Visualizer.read_image(image_path)
        out_image_path = os.path.join(
            images_dir, "_q-{}_c{}.jpg".format(p_id, str(cam_id).zfill(2)))
        Visualizer.save_image(image, out_image_path)

    @staticmethod
    def save_gallery_images(images_dir, image_tuples, distances):
        for i, (image_path, p_id, cam_id) in enumerate(image_tuples):
            out_image_path = os.path.join(
                images_dir, "g-{:.3f}-{}_c{}.jpg".format(distances[i], p_id,
                                                         str(cam_id).zfill(2)))

            image = Visualizer.read_image(image_path)
            Visualizer.save_image(image, out_image_path)

    def visualize_ranked_results(self, q_image_tuple, g_image_tuples, width=128,
                                 height=256, save_dir=''):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        qimg_path, qpid, qcamid = q_image_tuple
        # qimg_path = qimg_paths[0]

        qimg = cv.imread(qimg_path)
        qimg = cv.resize(qimg, (width, height))
        qimg = cv.copyMakeBorder(
            qimg, BW, BW, BW, BW, cv.BORDER_CONSTANT, value=(0, 0, 0)
        )
        # resize twice to ensure that the border width is consistent
        qimg = cv.resize(qimg, (width, height))
        num_cols = self.k + 1
        grid_img = 255 * np.ones(
            (height, num_cols * width + self.k * GRID_SPACING + Q_SPACING, 3),
            dtype=np.uint8
        )
        grid_img[:, :width, :] = qimg

        rank_idx = 1
        no_matches = True
        for g_image_tuple in g_image_tuples:
            gimg_path, gpid, gcamid = g_image_tuple
            # gimg_path = gimg_paths[0]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            # if not invalid:
            matched = gpid == qpid
            if matched and not invalid:
                no_matches = False
            border_color = GREEN if matched else RED
            gimg = cv.imread(gimg_path)
            gimg = cv.resize(gimg, (width, height))
            gimg = cv.copyMakeBorder(
                gimg, BW, BW, BW, BW, cv.BORDER_CONSTANT,
                value=border_color
            )
            gimg = cv.resize(gimg, (width, height))
            start = rank_idx * width + rank_idx * GRID_SPACING + Q_SPACING
            end = (rank_idx + 1
                   ) * width + rank_idx * GRID_SPACING + Q_SPACING
            grid_img[:, start:end, :] = gimg

            rank_idx += 1

        if no_matches:
            # abs_save_dir = os.path.abspath(save_dir)
            # os.rename(abs_save_dir, "{}-{}".format(abs_save_dir, 'f'))
            # print(save_dir)
            pass
        imname = osp.basename(osp.splitext(qimg_path)[0])
        cv.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

    def run(self, dist_mat, num=None):
        num_query, num_gallery = dist_mat.shape

        print("Saving images...")
        if num is not None:
            _num = min(num_query, num)
        else:
            _num = num_query

        for i in tqdm(range(_num)):
            # Get the k closest identities
            distances = np.array([dist_mat[i, j] for j in range(num_gallery)])
            idx = np.argpartition(distances, self.k)[:self.k]
            min_k = idx[np.argsort(distances[idx])]

            # Create a directory to store the images of the top k matches
            query_pid, query_cam_id = self.q_dataset[i][1:]
            out_pid_dir = os.path.join(
                self.out_dir, "{}_c{}".format(str(query_pid).zfill(6),
                                              str(query_cam_id).zfill(2)))
            if not os.path.isdir(out_pid_dir):
                os.makedirs(out_pid_dir)

            result_image_tuples = []
            result_distances = []
            for min_idx in min_k:
                result_distances.append(distances[min_idx])
                result_image_tuples.append(self.g_dataset[min_idx])

            # Save images
            # print(self.q_dataset)
            self.save_query_images(out_pid_dir, self.q_dataset[i])
            self.save_gallery_images(out_pid_dir, result_image_tuples,
                                     result_distances)
            self.visualize_ranked_results(self.q_dataset[i],
                                          result_image_tuples,
                                          save_dir=out_pid_dir)
