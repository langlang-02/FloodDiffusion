"""Physical contracts: view round trips, causal padding, and metric units."""
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scripts.g1.common import collate
from vendor.momask_g1.g1_io import (model_features_to_feature_bank, local_features_to_model_features,
    rotation_matrix_to_6d, G1_BANK_LAYOUT)
from vendor.momask_g1.g1_metrics import metric_values, ground_truth_native
from models.vae_wan_1d import VAEWanModel


class G1Contracts(unittest.TestCase):
    def setUp(self):
        n = 17
        self.pose = np.zeros((n, 38), dtype=np.float32)
        self.pose[:, 0] = np.linspace(0, 1, n)
        self.pose[:, 2] = 0.8
        angles = np.column_stack((np.linspace(-0.2, 0.1, n), np.linspace(0.1, -0.2, n), np.linspace(0, 0.3, n)))
        self.pose[:, 3:9] = rotation_matrix_to_6d(Rotation.from_euler('xyz', angles).as_matrix())
        self.pose[:, 9:] = np.linspace(0, 0.1, n)[:, None]

    def test_cropped_round_trip_both_views(self):
        bank, anchors = model_features_to_feature_bank(self.pose)
        for rotation, dimension in [('root_roll_pitch_rot6d', 68), ('root_roll_pitch_sincos', 66)]:
            fields = ['joint_angles', 'joint_vel', rotation, 'root_yaw_vel', 'root_xy_vel_body', 'root_height']
            indices = [i for field in fields for i in range(*G1_BANK_LAYOUT[field])]
            self.assertEqual(len(indices), dimension)
            reconstructed = local_features_to_model_features(bank[3:, indices], fields, anchors[3])
            np.testing.assert_allclose(reconstructed, self.pose[3:], atol=2e-6)
        np.testing.assert_allclose(ground_truth_native(bank, anchors), self.pose, atol=2e-6)

    def test_metric_units_and_root_alignment(self):
        gt = self.pose.astype(np.float64)
        pred = gt.copy(); pred[:, 0] += 1
        points = np.zeros((len(gt), 29, 3))
        shifted = points.copy(); shifted[:, :, 0] = 1
        values = metric_values(gt, pred, points, shifted, 20, [0, 1, 2, 3])
        np.testing.assert_allclose(values['global_mpjpe_mm'], 1000)
        np.testing.assert_allclose(values['root_relative_mpjpe_mm'], 0, atol=1e-6)

    def test_padding_and_codec_length(self):
        torch.set_num_threads(2)
        model = VAEWanModel(input_dim=68, z_dim=4, dim=8, dec_dim=8, vel_window=[29, 58]).eval()
        for n in [4, 5, 8, 9, 64, 65]:
            batch = collate([{'feature': np.zeros((n, 68), np.float32)}])
            self.assertEqual(batch['feature'].shape[1] % 4, 1)
            with torch.no_grad():
                out = model.decode(model.encode(batch['feature']))
            self.assertEqual(out.shape, batch['feature'].shape)
            self.assertGreaterEqual(out.shape[1], n)

    def test_encoder_is_causal(self):
        torch.set_num_threads(2)
        model = VAEWanModel(input_dim=66, z_dim=4, dim=8, dec_dim=8, vel_window=[29, 58]).eval()
        x = torch.randn(1, 17, 66)
        future_changed = x.clone(); future_changed[:, 9:] += 10
        with torch.no_grad():
            first, second = model.encode(x), model.encode(future_changed)
        torch.testing.assert_close(first[:, :3], second[:, :3])


if __name__ == '__main__':
    unittest.main()
