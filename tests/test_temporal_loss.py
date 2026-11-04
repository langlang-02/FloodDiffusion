import sys,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.common import rotation_matrix_to_6d
from g1.temporal_loss import angular_velocity,temporal_terms
from g1.temporal_ddp import denominators,KEYS
class TemporalTests(unittest.TestCase):
 def native(self):
  g=np.zeros((1,9,38));g[:,:,2]=.8;g[0,:,3:9]=rotation_matrix_to_6d(Rotation.from_euler('z',np.linspace(0,.4,9)[:,None]).as_matrix());return torch.tensor(g,dtype=torch.float64)
 def test_identity_gradient_and_scipy_parity(self):
  angles=np.linspace(0,.4,9);r=Rotation.from_euler('z',angles[:,None]).as_matrix();x=torch.tensor(r[None],requires_grad=True)
  got=angular_velocity(x);ref=Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).as_rotvec()*20
  np.testing.assert_allclose(got.detach()[0],ref,atol=1e-10)
  eye=torch.eye(3,dtype=torch.float64).expand(1,5,3,3).clone().requires_grad_();v=angular_velocity(eye);v.square().sum().backward();self.assertTrue(torch.isfinite(eye.grad).all())
 def test_matches_moving_gt_not_zero_target(self):
  g=self.native();terms=temporal_terms(g,g,torch.tensor([9]));self.assertTrue(all(float(x)==0 for x in terms.values()))
  p=g.clone();p[:,:,3:9]=g[:,:1,3:9];self.assertGreater(float(temporal_terms(p,g,torch.tensor([9]))['root_rot_vel']),0)
 def test_padding_and_short_clips(self):
  g=self.native();p=g.clone();p[:,5:]+=10
  terms=temporal_terms(p,g,torch.tensor([5]));self.assertTrue(all(float(x)==0 for x in terms.values()))
  for n in (1,2):
   terms=temporal_terms(g[:,:n],g[:,:n],torch.tensor([n]));self.assertTrue(all(torch.isfinite(x) and float(x)==0 for x in terms.values()))
 def test_oscillation_gradient(self):
  g=self.native();p=g.clone();p[0,::2,9]+=.05;p.requires_grad_();loss=temporal_terms(p,g,torch.tensor([9]))['dof_acc'];loss.backward();self.assertGreater(float(loss),0);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum().item(),0)
 def test_pair_triple_denominators(self):
  d=denominators(dict(length=torch.tensor([1,2,5]),eligible=torch.tensor([True,False,True])))
  values=dict(zip(KEYS,d.tolist()));self.assertEqual(values['root_rot_vel'],5);self.assertEqual(values['dof_acc'],3);self.assertEqual(values['root_rot_acc'],3)
if __name__=='__main__':unittest.main()
