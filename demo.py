import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt

from core.pose_prior import PosePrior
from core.functions import estimate_z
from visualization.basic import plot_3d

if __name__ == "__main__":

    test_pose = scio.loadmat("constants/testPose.mat")
    test_kpts = np.array(test_pose['S1'])
    test_edges = np.array(test_pose['edges'])-1

    part_names = ['back-bone', 'R-shldr', 'R-Uarm', 'R-Larm', 'L-shldr', 'L-Uarm', 'L-Larm', 'head', \
    'R-hip', 'R-Uleg', 'R-Lleg', 'R-feet', 'L-hip', 'L-Uleg', 'L-Lleg', 'L-feet']

    N = len(part_names)

    pose_prior = PosePrior()
    flags = pose_prior.is_valid(test_kpts)

    torso_bones = [0, 1, 4, 8, 12]
    n_torso = np.setdiff1d(list(range(N)), torso_bones)
    rand_signs = np.sign(np.random.rand(N) - 0.5)

    d2 = test_kpts[:, test_edges[:, 0]] - test_kpts[:, test_edges[:, 1]]
    d2[0, n_torso] = rand_signs[n_torso]*abs(d2[0, n_torso])
    sd = estimate_z(d2, test_edges.T, np.zeros(3).T)

    flags_sd = pose_prior.is_valid(sd)

    invalid_indices = np.argwhere(flags_sd[0, :] == 0).reshape(-1)

    print("List of invalid bones")
    print([part_names[i] for i in invalid_indices])

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, projection='3d',
                         xlim=(-20, 20), ylim=(-20, 20), zlim=(-20, 20))
    bx = fig.add_subplot(122, projection='3d',
                         xlim=(-20, 20), ylim=(-20, 20), zlim=(-20, 20))


    plot_3d(test_kpts, test_edges, ax)
    plot_3d(sd, test_edges, bx)

    plt.show()