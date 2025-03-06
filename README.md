# insar-syn-gen
Pythonic generation of synthetic wrapped interferograms from long term velocity rates obtained from InSAR displacement time series. To be used as a machine learning ready finetuning dataset for large pretrained self-supervised (SSL) image models.

This repository is heavily based off two research papers from Anantrasirichai et al., 2019 and Anantrasirichai et al. 2021 (see references at the bottom of this README). The GitHub repository and matlab code for the first paper can be found at:

https://github.com/pui-nantheera/Synthetic_InSAR_image/tree/main

# Usage

## Configuration

A toml file can be found under `configs/insar_synthetic_vel.toml` which can be edited to change arguments of the below stages.

## Spatial variogram

`variogram/spatial_variogram.py` will create a csv file of variogram and covariance function parameters from full frame SatSense InSAR velocity measurements. Parameters are estimated for geoclipped subsets within the full frame (4.5km x 4.5km). Each subset has been detrended with a 1d polynomial before the experimental variogram is fit. Output plots are also generated for each subset for inspection.

![inspect_variogram_081A_12821_NZ_GNS_hres h5_lat_-38 878593_-38 833595_long_176 0405_176 0855](https://github.com/user-attachments/assets/976cffd8-c634-46d7-82c5-23cac41a38bd)

## Turbulent atmospheric noise

`turbulent_atm/gen_turbulent_noise.py` will use the csv generated from the spatial variogram stage to read in parameters for the maximum covariances and spatial decay constants for each covariance function per spatial subset. The range of these parameters is then used as an upper/lower bound for sampling new maxmimum covariances and spatial decay constants from a uniform distribution. These are then used to produce distance/velocity covariance matrices of the turbulent atmospheric noise field (as assumption is made of no stratified noise) which are further decomposed with Cholesky decomposition. The upper triangular Cholesky matrix can then be used to generate new correlated noise samples by mutplication with an uncorrelated random vector sampled from a standard normal distribution. Both unwrapped and wrapped turbulent noise samples are generated.

![turb_maxcov_2 146751542480711_decay_0 2341718236406554_21](https://github.com/user-attachments/assets/959255eb-06ce-4c20-a335-01cef17e5d81)

## Deformation source model

`deformation/gen_deformation.py` will use a Mogi source model to generate samples of deformation of varying satellite heading, incidence angle, volume change, and depth. Both unwrapped and wrapped (min-max normalized) deformation samples are generated.

![incidence_33_heading_60_vol-1e3 1_depth_0 1](https://github.com/user-attachments/assets/4ac8de2f-7bb3-4812-9716-6f17d7e7d490)

## Spike noise and decoherence estimation

`noise_decoherence_map/gen_noise_decoh.py` uses a 3x3 pixel median filter on 224x244 pixel slices of the original SatSense velocity grid `Y`. The filtered output `Y'` can then be used to generate and save a noise map samples `N` where `N = Y - Y'`. Decoherence is estimated using existing masked pixels (`M`) within the 224x224 pixel slices in the original SatSense velocity data that have been deemed unreliable, we make sure not to save masks that have too high a percentage of masked values (masks with 65% or greater masked pixels are not saved).

## Combining data

`combine_DT.py` combines the deformation (`D`) and turbulent atmospheric noise (`T`) as well as the random noise maps (`N`) and decoherence masks (`M`). All components are converted to radians (multiples of Sentinel-1 half wavelength, 2.8333cm) before summation so wrapped samples can be generated. Two classes of training data are produced: class `0` (set1) contains only `T+noise+dechorence` (no deformation) whereas class `1` (set2) contains `D+T+noise+decoherence`.

Unwrapped class 1 (D+T+noise+decoherence)             |  Wrapped class 1 (D+T+noise+decoherence)
:-------------------------:|:-------------------------:
![unwrapped_DT_incidence_33_heading_60_vol-1e3 1_depth_0 1_turb_maxcov_8 11867555858447_decay_1 098962094497368_142](https://github.com/user-attachments/assets/ff57881c-e3f1-443a-b971-7d9de85d7117) | ![wrapped_DT_incidence_33_heading_60_vol-1e3 1_depth_0 1_turb_maxcov_8 11867555858447_decay_1 098962094497368_142](https://github.com/user-attachments/assets/0fe11a4d-cd39-43c2-a314-e6a07be0ab64)

## Interpolation

`interpolation/run_interp.py` will run Delaunay Triangulation and linear interpolation within triangles (edges) only under a certain size/threshold. This is to ensure interpolation is not performed accross large distances.

Unwrapped Interpolation with Delaunay Triangulation (class 1) | Wrapped Interpolation with Delaunay Triangulation (class 1)
:-------------------------:|:-------------------------:
![example_interpolated_image](https://github.com/user-attachments/assets/b2884ff9-2a13-41d4-9d48-aa84d0efefd1) | ![example_interpolated_image_wrapped](https://github.com/user-attachments/assets/7b7722a5-cf39-485c-8add-16ea7d202c8c)

## BibTeX references
```
@ARTICLE{9181454,
  author={Anantrasirichai, Nantheera and Biggs, Juliet and Kelevitz, Krisztina and Sadeghi, Zahra and Wright, Tim and Thompson, James and Achim, Alin Marian and Bull, David},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Detecting Ground Deformation in the Built Environment Using Sparse Satellite InSAR Data With a Convolutional Neural Network}, 
  year={2021},
  volume={59},
  number={4},
  pages={2940-2950},
  keywords={Strain;Satellites;Convolution;Machine learning;Training data;Terrain factors;Velocity measurement;Convolutional neural network (CNN);earth observation;ground deformation;interferometric synthetic aperture radar (InSAR);machine learning},
  doi={10.1109/TGRS.2020.3018315}}
```

```
@article{ANANTRASIRICHAI2019111179,
  title = {A deep learning approach to detecting volcano deformation from satellite imagery using synthetic datasets},
  journal = {Remote Sensing of Environment},
  volume = {230},
  pages = {111179},
  year = {2019},
  issn = {0034-4257},
  doi = {https://doi.org/10.1016/j.rse.2019.04.032},
  url = {https://www.sciencedirect.com/science/article/pii/S003442571930183X},
  author = {N. Anantrasirichai and J. Biggs and F. Albino and D. Bull},
  keywords = {Interferometric Synthetic Aperture Radar, Volcano, Machine learning, Detection}}
```


