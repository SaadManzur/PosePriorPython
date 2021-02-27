# Ongoing Pythonic Implementation of "Pose-Conditioned Joint Angle Limits for 3D Human Pose Reconstruction"

Modules implemented:
* Validation check

Please cite the original paper

```
@inproceedings{Akhter:CVPR:2015,
  title = {Pose-Conditioned Joint Angle Limits for {3D} Human Pose Reconstruction},
  author = {Akhter, Ijaz and Black, Michael J.},
  booktitle = { IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) 2015},
  month = jun,
  abstract = {The estimation of 3D human pose from 2D joint locations is central to many vision problems involving the analysis
  of people in images and video. To address the fact that the problem is inherently ill posed, many methods impose
  a prior over human poses. Unfortunately these priors admit invalid poses because they do not model how joint-limits
  vary with pose. Here we make two key contributions. First, we collected a motion capture dataset that explores a wide
  range of human poses. From this we learn a pose-dependent model of joint limits that forms our prior. The dataset and
  the prior will be made publicly available. Second, we define a general parameterization of body pose and a new, multistage, method to estimate 3D pose from 2D joint locations that uses an over-complete dictionary of human poses. Our method shows good generalization while avoiding impossible poses. We quantitatively compare our method with
  recent work and show state-of-the-art results on 2D to 3D pose estimation using the CMU mocap dataset. We also
  show superior results on manual annotations on real images and automatic part-based detections on the Leeds sports
  pose dataset.},
  year = {2015}
}
```