import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tune_mask_hough import candidates,fit_centers


class CenterpointTests(unittest.TestCase):
    def test_bars_with_occlusion(self):
        image=np.zeros((360,640),np.uint8)
        for x in (100,250,400):
            cv2.fillConvexPoly(image,np.array([[x-8,0],[x+8,0],[x+23,359],[x+7,359]]),255)
        image[150:190]=0
        result=fit_centers(candidates(image),image.shape,dict(merge=16,band=8,support=.25))
        self.assertEqual(len(result),3)
        np.testing.assert_allclose(result,[107,257,407],atol=3)

    def test_empty_and_horizontal(self):
        image=np.zeros((360,640),np.uint8)
        image[100:120]=255
        self.assertEqual(fit_centers(candidates(image),image.shape,dict(merge=16,band=8,support=.25)),[])


if __name__=='__main__':
    unittest.main()
