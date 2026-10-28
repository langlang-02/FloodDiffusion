"""Differentiable FK from compiled MuJoCo transforms; no dynamics or clipping."""
import numpy as np
import mujoco
import torch
from torch import nn
from torch.nn import functional as F
from .common import JOINT_NAMES


def rot6d(value):
    first = value[..., :3]
    norm = first.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(first)
    fallback[..., 0] = 1
    first = torch.where(norm > 1e-8, first / norm.clamp_min(1e-8), fallback)
    second = value[..., 3:] - (value[..., 3:] * first).sum(-1, keepdim=True) * first
    norm = second.norm(dim=-1, keepdim=True)
    reference = F.one_hot(first.abs().argmin(-1), 3).to(first)
    second = torch.where(norm > 1e-8, second / norm.clamp_min(1e-8),
                         torch.linalg.cross(first, reference))
    second = F.normalize(second, dim=-1, eps=1e-8)
    return torch.stack((first, second, torch.linalg.cross(first, second)), dim=-1)


def matrix_quaternion(m):
    """Stable branch selection, wxyz. Branch algebra avoids acos near identity."""
    a, b, c = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    squared = torch.stack((1+a+b+c, 1+a-b-c, 1-a+b-c, 1-a-b+c), -1)
    roots = squared.clamp_min(1e-12).sqrt()
    candidates = torch.stack((
        torch.stack((squared[..., 0], m[..., 2, 1]-m[..., 1, 2], m[..., 0, 2]-m[..., 2, 0], m[..., 1, 0]-m[..., 0, 1]), -1),
        torch.stack((m[..., 2, 1]-m[..., 1, 2], squared[..., 1], m[..., 1, 0]+m[..., 0, 1], m[..., 0, 2]+m[..., 2, 0]), -1),
        torch.stack((m[..., 0, 2]-m[..., 2, 0], m[..., 1, 0]+m[..., 0, 1], squared[..., 2], m[..., 2, 1]+m[..., 1, 2]), -1),
        torch.stack((m[..., 1, 0]-m[..., 0, 1], m[..., 0, 2]+m[..., 2, 0], m[..., 2, 1]+m[..., 1, 2], squared[..., 3]), -1),
    ), -2) / (2 * roots[..., :, None].clamp_min(.1))
    index = squared.argmax(-1)[..., None, None].expand(*squared.shape[:-1], 1, 4)
    return F.normalize(candidates.gather(-2, index).squeeze(-2), dim=-1)


class TorchG1FK(nn.Module):
    def __init__(self, mjcf):
        super().__init__()
        m = mujoco.MjModel.from_xml_path(str(mjcf))
        self.parents = m.body_parentid.tolist()
        self.body_joint = []
        joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        if min(joint_ids) < 0:
            raise ValueError('MJCF must contain all 29 named G1 hinge joints')
        self.joint_bodies = m.jnt_bodyid[joint_ids].tolist()
        free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError('Exactly one free joint required')
        self.root_body = int(m.jnt_bodyid[free[0]])
        if self.parents[self.root_body] != 0:
            raise ValueError('Free root must be a child of world')
        self.loss_bodies = [self.root_body] + self.joint_bodies
        for body in range(m.nbody):
            count = int(m.body_jntnum[body])
            if count > 1:
                raise ValueError('Multiple joints per body are unsupported')
            jid = int(m.body_jntadr[body]) if count else -1
            if jid in joint_ids:
                self.body_joint.append(joint_ids.index(jid))
            elif jid >= 0 and body != self.root_body:
                raise ValueError(f'Unexpected joint on body {body}')
            else:
                self.body_joint.append(-1)
        rotations = []
        for q in m.body_quat:
            r = np.empty(9)
            mujoco.mju_quat2Mat(r, q)
            rotations.append(r.reshape(3, 3))
        for name, value in dict(offset=m.body_pos, fixed_rotation=np.array(rotations),
                                joint_pos=m.jnt_pos[joint_ids], axis=m.jnt_axis[joint_ids],
                                reference=m.qpos0[m.jnt_qposadr[joint_ids]]).items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.feet = [JOINT_NAMES.index(n) for n in ('left_ankle_roll_joint', 'right_ankle_roll_joint')]

    def forward(self, native):
        shape = native.shape[:-1]
        identity = torch.eye(3, device=native.device, dtype=native.dtype).expand(*shape, 3, 3)
        positions = [native.new_zeros(*shape, 3)]
        rotations = [identity]
        anchors = [None] * len(JOINT_NAMES)
        for body in range(1, len(self.parents)):
            parent = self.parents[body]
            if body == self.root_body:
                position, rotation = native[..., :3], rot6d(native[..., 3:9])
            else:
                rotation = rotations[parent] @ self.fixed_rotation[body]
                position = positions[parent] + (rotations[parent] @ self.offset[body, :, None]).squeeze(-1)
                joint = self.body_joint[body]
                if joint >= 0:
                    pivot = self.joint_pos[joint]
                    anchor = position + (rotation @ pivot[:, None]).squeeze(-1)
                    anchors[joint] = anchor
                    axis = self.axis[joint]
                    x, y, z = axis.unbind()
                    zero = x * 0
                    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
                    angle = native[..., 9 + joint] - self.reference[joint]
                    delta = identity + angle.sin()[..., None, None] * skew + (1-angle.cos())[..., None, None] * (skew @ skew)
                    rotation = rotation @ delta
                    position = anchor - (rotation @ pivot[:, None]).squeeze(-1)
            positions.append(position)
            rotations.append(rotation)
        return dict(anchors=torch.stack(anchors, -2),
                    body_pos=torch.stack([positions[i] for i in self.loss_bodies], -2),
                    body_rot=torch.stack([rotations[i] for i in self.loss_bodies], -3))
