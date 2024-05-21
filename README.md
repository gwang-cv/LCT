# LCT

Pytorch implementation of LCT for ICME 2023 paper "[Local Consensus Transformer for Correspondence Learning](https://ieeexplore.ieee.org/abstract/document/10219942/)", by Gang Wang and Yufei Chen.

The [pretrained models](https://drive.google.com/drive/folders/1N3YFwUt3w7ttDCxHKlzmOeWMjOQdTarG?usp=drive_link) can be downloaded and saved in the 'model' folder, including 'yfcc-sift', 'yfcc-superpoint', 'sun3d-sift', and 'sun3d-superpoint'.



If you find this project useful, please cite:

	@inproceedings{wang2023local,
	  title={Local Consensus Transformer for Correspondence Learning},
	  author={Wang, Gang and Chen, Yufei},
	  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
	  pages={1151--1156},
	  year={2023},
	  organization={IEEE}
	}


Requirements
-
Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.


Acknowledgement
-
This code is heavily borrowed from [OANet](https://github.com/zjhthu/OANet). If you use the part of code related to data generation, testing and evaluation, you should cite this paper and follow its license.


	@article{zhang2019oanet,
	  title={Learning Two-View Correspondences and Geometry Using Order-Aware Network},
	  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
	  journal={International Conference on Computer Vision (ICCV)},
	  year={2019}
	}

