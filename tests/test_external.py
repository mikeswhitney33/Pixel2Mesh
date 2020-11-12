import unittest
from p2m_utils import path_utils
import tensorflow as tf
import os.path as osp



class InstallTest(unittest.TestCase):
    def test_nn_distance(self):
        # base_dir = osp.dirname(osp.abspath(__file__))
        nn_distance_module = tf.load_op_library(osp.join(path_utils.get_external_dir(), 'tf_nndistance_so.so'))
        self.assertTrue(hasattr(nn_distance_module, "nn_distance"))


if __name__ == "__main__":
    unittest.main()
