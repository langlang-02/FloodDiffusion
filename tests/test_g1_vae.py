"""Behavioral checks: MuJoCo parity, causal boundary, padding, and loss masking."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1.common import G1Kinematics, rotation_matrix_to_6d
from g1.data import contact_policy, source_group, split_for_group
from g1.kinematics import TorchG1FK
from g1.model import G1CausalVAE, TextOpLoss
from scipy.spatial.transform import Rotation


class G1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cls.xml = ROOT/'outputs/g1_smoke_data/kinematic.xml'
        if not cls.xml.exists():
            raise RuntimeError('Prepare outputs/g1_smoke_data before running tests')

    def model(self):
        return G1CausalVAE(np.zeros(38,np.float32),np.ones(38,np.float32),
                          z_dim=4,dim=8,dec_dim=8,num_res_blocks=1,
                          dim_mult=[1,1,1],temperal_downsample=[True,True]).eval()

    def native(self, length=13):
        rng = np.random.default_rng(7)
        x = rng.normal(0,.1,(length,38)).astype(np.float32)
        x[:,2] += .8
        x[:,3:9] = rotation_matrix_to_6d(Rotation.random(length,random_state=rng).as_matrix())
        return x

    def test_fk_matches_mujoco_and_gradients(self):
        x = self.native()
        torch_fk = TorchG1FK(self.xml)
        tensor = torch.tensor(x, requires_grad=True)
        actual = torch_fk(tensor)
        expected = G1Kinematics(self.xml).positions(x)
        np.testing.assert_allclose(actual['anchors'].detach().numpy(), expected, atol=4e-7, rtol=2e-5)
        actual['anchors'].square().mean().backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(tensor.grad[:,9:].abs().sum().item(), 0)

    def test_causal_encoder_and_stream_parity(self):
        model = self.model()
        a = torch.tensor(self.native(17))[None]
        b = a.clone()
        b[:,9:] += torch.randn_like(b[:,9:]) * 2
        with torch.no_grad():
            za = model.encode(a)
            zb = model.encode(b)
            # Latents at timestamps 0,4,8 cannot depend on frames 9 onward.
            torch.testing.assert_close(za[:,:3],zb[:,:3],atol=0,rtol=0)
            model.clear_cache()
            zstream = torch.cat((model.stream_encode(a[:,:1],True),
                                 model.stream_encode(a[:,1:9],False),
                                 model.stream_encode(a[:,9:],False)),dim=1)
            torch.testing.assert_close(za,zstream,atol=1e-6,rtol=1e-5)
            model.clear_cache()
            streamed = torch.cat((model.stream_decode(za[:,:1],True),
                                  model.stream_decode(za[:,1:3],False),
                                  model.stream_decode(za[:,3:],False)),dim=1)
            torch.testing.assert_close(model.decode(za),streamed,atol=1e-6,rtol=1e-5)

    def test_tail_is_kept(self):
        model=self.model()
        for length in (1,2,4,5,8,13,16,17):
            x=torch.tensor(self.native(length))[None]
            with torch.no_grad():
                out,_,_=model.reconstruct(x)
            self.assertEqual(out.shape,x.shape)
            self.assertTrue(torch.isfinite(out).all())

    def test_contact_exclusion_has_zero_gradient(self):
        model=self.model()
        loss_fn=TextOpLoss(TorchG1FK(self.xml))
        gt=torch.tensor(self.native(9))[None]
        pred=(gt+.02).requires_grad_()
        mu=torch.zeros(1,4,3)
        logvar=torch.zeros_like(mu)
        lengths=torch.tensor([9])
        contact=torch.ones(1,9,2,dtype=torch.bool)
        losses=loss_fn(model,pred,gt,mu,logvar,lengths,contact,torch.tensor([False]))
        self.assertEqual(set(losses), {'rec','kl','body_trans','body_rot','dof_pos','dof_vel','foot_contact','total'})
        self.assertEqual(losses['foot_contact'].item(),0)
        grad=torch.autograd.grad(losses['foot_contact'],pred,retain_graph=True)[0]
        self.assertEqual(grad.abs().sum().item(),0)
        losses['total'].backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        enabled=loss_fn(model,pred,gt,mu,logvar,lengths,contact,torch.tensor([True]))
        self.assertGreater(enabled['foot_contact'].item(),0)
        identity=loss_fn(model,gt,gt,mu,logvar,lengths,contact,torch.tensor([True]))
        self.assertAlmostEqual(identity['total'].item(),0,places=7)

    def test_padding_does_not_change_losses(self):
        model = self.model()
        loss_fn = TextOpLoss(TorchG1FK(self.xml))
        gt = torch.tensor(self.native(13))[None].repeat(2, 1, 1)
        pred = gt + .03
        lengths = torch.tensor([5, 9])
        contact = torch.ones(2, 13, 2, dtype=torch.bool)
        eligible = torch.ones(2, dtype=torch.bool)
        mu = torch.zeros(2, 4, 4)
        logvar = torch.zeros_like(mu)
        expected = loss_fn(model, pred, gt, mu, logvar, lengths, contact, eligible)
        changed = pred.clone()
        changed[0, 5:] += 10
        changed[1, 9:] -= 10
        mu[0, :, 2:] = 7
        mu[1, :, 3:] = 7
        actual = loss_fn(model, changed, gt, mu, logvar, lengths, contact, eligible)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_terrain_and_grouping(self):
        for caption in ('walking downstairs','going up a staircase','walking on a ramp','climbing a ladder'):
            self.assertFalse(contact_policy({'content_short_description':caption})[0])
        for caption in ('step sideways and walk forward', 'jump up and land lightly', 'start walking, stepping off with the right foot'):
            self.assertTrue(contact_policy({'content_short_description':caption})[0])
        self.assertFalse(contact_policy({'content_short_description':'step down from a 50cm box'})[0])
        row=dict(filename='motion',take_date=20200101,take_actor='A001',take_org_name='walk')
        mirror=dict(row,filename='motion_M',take_org_name='walk_M')
        self.assertEqual(source_group(row),source_group(mirror))
        self.assertEqual(split_for_group(source_group(row)),split_for_group(source_group(mirror)))


if __name__=='__main__':
    unittest.main()
