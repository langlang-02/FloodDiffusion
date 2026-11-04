"""Absolute yaw must drive orientation, survive crops, and backpropagate correctly."""
import sys,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.rot6d69 import encode_native,decode_native
from g1.rot6d68 import encode_native as encode68
from g1.common import rotation_matrix_to_6d
class Rot69Tests(unittest.TestCase):
 def setUp(self):
  t=np.linspace(0,1,41);g=np.zeros((41,38),np.float32);g[:,:3]=np.stack((t,t*t,.8+t*.01),axis=1)
  g[:,3:9]=rotation_matrix_to_6d(Rotation.from_euler('ZYX',np.stack((t*4*np.pi,t*.1,t*.05),axis=1)).as_matrix());g[:,9:]=np.sin(t[:,None]+np.arange(29))*.2
  self.g=g;self.x,self.a=encode_native(g)
 def test_preserved68_and_unwrapped_yaw(self):
  np.testing.assert_array_equal(self.x[:,:68],encode68(self.g)[0]);self.assertGreater(self.x[-1,68],12)
  self.assertLess(np.abs(np.diff(self.x[:,68])).max(),.4)
 def test_full_and_crop_roundtrip(self):
  for start in (0,17,29):
   y=decode_native(torch.tensor(self.x[None,start:]),torch.tensor(self.a[start:start+1],dtype=torch.float32))[0].numpy()
   np.testing.assert_allclose(y,self.g[start:],atol=3e-6,rtol=3e-6)
 def test_no_yaw_delta_integration_or_gt_yaw(self):
  x=torch.tensor(self.x[None]);a=torch.tensor(self.a[:1],dtype=torch.float32);expected=decode_native(x,a)
  x[:,:,64]+=10;a[:,3]+=20
  torch.testing.assert_close(decode_native(x,a),expected,atol=0,rtol=0)
 def test_absolute_yaw_gradient(self):
  x=torch.tensor(self.x[None],dtype=torch.float64,requires_grad=True);a=torch.tensor(self.a[:1],dtype=torch.float64)
  y=decode_native(x,a)[0,-1,:9].square().sum()+decode_native(x,a)[0,-1,3];y.backward()
  idx=(0,23,68);plus=x.detach().clone();minus=plus.clone();plus[idx]+=1e-5;minus[idx]-=1e-5
  def score(q):
   z=decode_native(q,a);return z[0,-1,:9].square().sum()+z[0,-1,3]
  torch.testing.assert_close(x.grad[idx],(score(plus)-score(minus))/2e-5,atol=1e-7,rtol=1e-4)
  self.assertGreater(x.grad[...,68].abs().sum().item(),0);self.assertEqual(x.grad[...,64].abs().sum().item(),0)
if __name__=='__main__':unittest.main()
