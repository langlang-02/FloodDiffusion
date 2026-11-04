"""Root temporal errors in physical units, independent of spatial MPJPE."""
import numpy as np
from scipy.spatial.transform import Rotation
from .common import rotation_6d_to_matrix

def temporal_values(gt,pred,fps=20.):
    def velocity(x):
        r=rotation_6d_to_matrix(x[:,3:9])
        return Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).as_rotvec()*fps
    gv,pv=velocity(gt),velocity(pred)
    return dict(root_angular_velocity_rmse_rad_s=pv-gv,
                root_angular_acceleration_rmse_rad_s2=np.diff(pv,axis=0)*fps-np.diff(gv,axis=0)*fps)
