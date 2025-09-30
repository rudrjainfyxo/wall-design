import tensorflow as tf
import cv2
import tarfile
import numpy as np

LABEL_WALL = 12  # ADE20K label for "wall"
MODEL_TAR = "deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz"

class DeepLabModel:
    def __init__(self, tarball_path=MODEL_TAR):
        self.graph = tf.Graph()

        with tarfile.open(tarball_path) as tar:
            for member in tar.getmembers():
                if "frozen_inference_graph" in member.name:
                    file = tar.extractfile(member)
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(file.read())
                    break

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.INPUT_TENSOR_NAME = 'ImageTensor:0'
        self.OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    def predict(self, bgr_image):
        image = cv2.resize(bgr_image, (513, 513))
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={
            self.INPUT_TENSOR_NAME: [image[..., ::-1]]  # BGR to RGB
        })
        seg_map = batch_seg_map[0]
        seg_map = cv2.resize(seg_map.astype(np.uint8), (bgr_image.shape[1], bgr_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return seg_map

def get_wall_mask_midas(image, depth, min_thresh=0.2, max_thresh=0.6):
    model = DeepLabModel()
    seg_map = model.predict(image)
    wall_mask = (seg_map == LABEL_WALL).astype(np.uint8) * 255

    norm_depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    depth_mask = ((norm_depth > min_thresh) & (norm_depth < max_thresh)).astype(np.uint8) * 255

    combined = cv2.bitwise_and(wall_mask, depth_mask)
    return combined
