import unittest
import numpy as np
from spatial_check import query_grid, metrics

class SpatialTests(unittest.TestCase):
    def test_grid_order(self):
        np.testing.assert_array_equal(query_grid([10,20],[0,.5,1]),[[10,0],[10,.5],[10,1],[20,0],[20,.5],[20,1]])
    def test_time_change_is_not_spatial_range(self):
        T=np.array([[150,160,155],[170,180,175]])
        m=metrics(T,np.array([[.5,.6,.55],[.8,.9,.85]]))
        np.testing.assert_allclose(m['delta_T'],[10,10])
        np.testing.assert_allclose(m['delta_alpha'],[.1,.1])
        self.assertEqual(T.max()-T.min(),30)
    def test_uniform_space_changing_time(self):
        m=metrics(np.array([[20,20,20],[180,180,180]]),np.array([[.05]*3,[.9]*3]))
        np.testing.assert_array_equal(m['delta_T'],[0,0])
        np.testing.assert_array_equal(m['delta_alpha'],[0,0])
    def test_single_position_rejected(self):
        with self.assertRaises(ValueError): metrics(np.ones((4,1)),np.ones((4,1)))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): metrics([[1,np.nan]],[[.2,.3]])
    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError): metrics(np.ones((2,3)),np.ones((3,2)))

if __name__=='__main__': unittest.main()
