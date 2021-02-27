import numpy as np
import scipy.io as scio

from core.functions import get_normal
from core.functions import gram_schmidt
from core.functions import cart_to_spherical

class PosePrior(object):
    
    '''
    di: 
    a:         Arbitrary vector to generate local co-ordinate axes
    jmp:       Used for generating discretized bins for azim and elev
    childs:    Children excluding torso
    parents:   Parents excluding torso
    edges: 
    angleSprd: Angle spread limits (binary)
    sepPlane:  Seperating plane to further put limits on the child bones
    E2:        Stores normals to do a 2D projection for fine tuning the bounds
    bounds:    Actual bounds
    childsT:   Torso's childs
    '''
    
    def __init__(self):
        static_pose = scio.loadmat("constants/staticPose.mat")
        self.di = static_pose['di']
        self.a = static_pose['a']

        joint_angle_model_v2 = scio.loadmat("constants/jointAngleModel_v2.mat")
        self.jmp = joint_angle_model_v2['jmp']
        self.childs = joint_angle_model_v2['chlds'][0, :]-1
        self.parents = joint_angle_model_v2['prnts'][0, :]-1
        
        self.edges = joint_angle_model_v2['edges']-1
        self.angleSprd = joint_angle_model_v2['angleSprd'][0, :]
        self.sepPlane = joint_angle_model_v2['sepPlane'][0, :]
        self.E2 = joint_angle_model_v2['E2'][0, :]
        self.bounds = joint_angle_model_v2['bounds'][0, :]
        self.childsT = [2, 5, 7, 9, 13]
        self.a = [0.997478132770155, 0.002323490336655, 0.070936422506501]
        
    def is_valid(self, pose):
        
        n_parts = self.childs.shape[-1]
        
        parent_relative_pose = pose[:, self.edges[:, 0]] - pose[:, self.edges[:, 1]]
        
        flags = np.ones((1, parent_relative_pose.shape[-1]))
        angles = np.zeros((2, parent_relative_pose.shape[-1]))
        pose_local = self.global2local(parent_relative_pose)
        
        for i in range(n_parts):
            
            child_bone = pose_local[:, self.childs[i]]
            
            r, theta, phi = cart_to_spherical(child_bone)
            
            child_bone = child_bone / r
            theta = np.rad2deg(theta)
            phi = np.rad2deg(phi)
            
            t_j = int(np.floor((theta + 180)/self.jmp + 1))
            p_j = int(np.floor((phi + 90)/self.jmp + 1))
            
            angles[:, self.childs[i]] = [t_j, p_j]
            
            if self.childs[i] in self.childsT:
                
                if not self.angleSprd[i][t_j-1, p_j-1]:
                    flags[0, self.childs[i]] = False
            else:
                t_p = int(angles[0, self.parents[i]])
                p_p = int(angles[1, self.parents[i]])
                
                v = np.squeeze(self.sepPlane[i][t_p-1, p_p-1, :])
                
                v = v / np.linalg.norm(v[:3])
                
                child_bone_h = np.vstack((child_bone.reshape(-1, 1), np.ones((1, 1)))) 
                
                if np.isnan(v).any() or np.matmul(v.T, child_bone_h[:, 0]) > 0:
                    flags[0, self.childs[i]] = False
                else:
                    e1 = v[0:3]
                    e2 = np.squeeze(self.E2[i][t_p-1, p_p-1, :])
                    T = gram_schmidt(np.vstack([e1, e2, np.cross(e1, e2)]).T)
                    bnd = np.squeeze(self.bounds[i][t_p-1, p_p-1, :])
                    
                    u = np.matmul(T[:, 1:3].T, child_bone)
                    
                    if u[0] < bnd[0] or u[0] > bnd[1] or u[1] < bnd[2] or u[1] > bnd[3]:
                        flags[0, self.childs[i]] = False
                        
        return flags
        
        
    def global2local(self, pose, typ=1):
        
        n_parts = self.childs.shape[-1]
        
        shoulder = pose[:, 4] - pose[:, 1]
        hip = pose[:, 12] - pose[:, 8]
        
        pose_local = pose.copy()
        
        u, v, w, R = None, None, None, None
        
        for i in range(n_parts):
            
            if self.childs[i] in self.childsT:
                
                if i == 0 or i == 2 or i == 4:
                    u = shoulder
                else:
                    u = hip
                
                u = u / np.linalg.norm(u)
                v = pose[:, 0]
                v = v / np.linalg.norm(v)
            
            else:
                
                u = pose[:, self.parents[i]]
                u = u / np.linalg.norm(u)
                
                v, _ = get_normal(np.matmul(R,self.di[:, self.parents[i]]), np.matmul(R, self.a), u)
                
            w = np.cross(u, v)
            w = w / np.linalg.norm(w)
            
            if typ == 1:
                R = gram_schmidt(np.vstack([u, v, w]).T)
            elif typ == 2:
                R = gram_schmidt(np.vstack([w, u, v]).T)
            elif typ == 3:
                R = gram_schmidt(np.vstack([v, w, u]).T)
            elif typ == 4:
                R = gram_schmidt(np.vstack([u, -w, v]).T)
            elif typ == 5:
                R = gram_schmidt(np.vstack([-w, v, u]).T)
            else:
                R = gram_schmidt(np.vstack([v, u, -w]).T)
            
            pose_local[:, self.childs[i]] = np.matmul(R.T, pose[:, self.childs[i]])
            
        return pose_local