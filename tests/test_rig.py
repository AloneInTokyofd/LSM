import base64
import types
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from config import FEATURE_DIM
from recognition_features import regions
from rig_overlay import rig_payload
from web_app import Runtime


class RigTests(unittest.TestCase):
    def test_coordinates_are_not_mirrored_twice_and_absent_points_are_hidden(self):
        points = np.zeros(FEATURE_DIM, dtype=np.float32)
        pose, face, left, _ = regions(points)
        pose[11] = [.2, .3, .01, .9]
        pose[12] = [.8, .3, .01, .1]
        left[0] = [.7, .6, .01]
        face[0] = [np.nan, .5, 0]
        result = rig_payload(points, 640, 480)
        self.assertAlmostEqual(result['left'][0][0], .7)
        self.assertIsNotNone(result['pose'][11])
        self.assertIsNone(result['pose'][12])
        self.assertIsNone(result['face'][0])
        self.assertTrue(all(p is None for p in result['right']))
        self.assertEqual((result['width'], result['height']), (640,480))

    def test_frame_returns_rig_only_when_requested_without_a_trained_model(self):
        runtime = Runtime()
        sid = runtime.new_session('lsm')
        runtime.session(sid).holistic = object()
        points = np.zeros(FEATURE_DIM, dtype=np.float32)
        regions(points)[2][0] = [.7,.6,.01]
        fake = types.SimpleNamespace(mediapipe_detection=lambda image, model:(image, object()),
                                     extract_keypoints=lambda result:points, hands_detected=lambda result:True)
        image = cv2.imencode('.jpg', np.zeros((16,32,3), np.uint8))[1].tobytes()
        payload = {'session':sid, 'image':base64.b64encode(image).decode(), 'mode':'train'}
        with patch.dict('sys.modules', {'utils':fake}):
            self.assertNotIn('rig', runtime.frame(payload))
            result = runtime.frame({**payload, 'rig':True})
            self.assertAlmostEqual(result['rig']['left'][0][0], .7)
            self.assertEqual(result['rig']['width'], 32)
            self.assertEqual(len(runtime.session(sid).frames), 0)
            self.assertIsNone(result['candidate'])
