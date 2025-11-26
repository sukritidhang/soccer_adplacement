# soccer_adplacement
This code is only for academic and research purposes.


## Code Organization

All codes are written in python3.


### Dependencies 

The following libraries should be installed before the execution of the codes



- numpy: pip install numpy
- pandas: pip install pandas
- matplotlib: pip install matplotlib
- glob: pip install glob
- scikit-image: pip install scikit-image

### Data

<p>The SoccerNet dataset [1], which provides robust annotations including field layouts, players positions, and event labels. For this study, we focus on the calibration subset, which includes line segment annotations essential for camera parameter estimation. Specifically, we automatically extract penalty area annotations, which are critical for accurate spatial localization and homography estimation. </p>

> Magera F, Hoyoux T, Barnich O, et al (2024) A universal protocol to benchmark cam-
era calibration for sports. In: IEEE International Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), CVsports, Seattle, Washington,
USA


### Scripts 
1. In the following folder structure inside ./polygon selction.
- python3 datasetvd.py: Run this script to automatically select a geometrically
consistent quadrilateral region inside the penalty area from field calibration
coordinates.
2. python3  ad_placement_video.py: Run this script for  homography-based warping. Integrates
instance-level occlusion masks from Mask R-CNN with Laplacian Alpha Blending so that the virtual advert is correctly placed behind players and the ball.
3. python3 occl_error_comp.py: Run this script to compute the occlusion error.