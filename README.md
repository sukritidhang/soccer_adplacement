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
2. python3  ad_placement_occl_video.py: Run this script for  homography-based warping. Integrates instance-level occlusion masks from Mask R-CNN with Laplacian Alpha Blending so that the virtual advert is correctly placed behind players and the ball.
3. python3 ad_placement_wo_occl_video.py: Run this script for homography-based warping. Integrates Laplacian Alpha Blending without considering the occlusion. The virtual advert is  placed over the players and the ball.
4. python3 occl_error_comp.py: Run this script to compute the occlusion error.
5. In the following folder structure inside ./metric.
    - python3 consistency_flicker_index_plt.py: Run this script to evalute the consistency over consecutive frames. Analyze temporal consistency using optical flow consistency and flicker index metrics. These metrics help determine whether player movements and overall video dynamics remain unaffected by the inserted advertisement.
    - python3 ssim_psnr.py: Run this script to evaluate the visual fidelity of our advert insertion framework, between the original and modified video frames