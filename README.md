# PPL (Paired-point lifting)
## Paired-Point Lifting for Enhanced Privacy-Preserving Visual Localization (CVPR 2023)

**Authors:** Chunghwan Lee, Jaihoon Kim, Chanhyuk Yun, [Je Hyeong Hong](https://sites.google.com/view/hyvision)

**[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lee_Paired-Point_Lifting_for_Enhanced_Privacy-Preserving_Visual_Localization_CVPR_2023_paper.pdf)][[Supplementary document](documents/Lee_et_al_cvpr23_supplemat.pdf)]**

- **Teaser videos for PPL**

https://github.com/Fusroda-h/ppl/assets/64474003/78b7ecb5-717e-44a6-96fb-aed6cb8a4a59


https://github.com/Fusroda-h/ppl/assets/64474003/167f1f00-7989-45a5-9f2f-896387ccde82


https://github.com/Fusroda-h/ppl/assets/64474003/aad1fc65-5f88-4930-a73a-2234cf09e7fc


**Abstract:** Visual localization refers to the process of recovering camera pose from input image relative to a known scene, forming a cornerstone of numerous vision and robotics systems. While many algorithms utilize sparse 3D point cloud of the scene obtained via structure-from-motion (SfM) for localization, recent studies have raised privacy concerns by successfully revealing high-fidelity appearance of the scene from such sparse 3D representation. One prominent approach for bypassing this attack was to lift 3D points to randomly oriented 3D lines thereby hiding scene geometry, but latest work have shown such random line cloud has a critical statistical flaw that can be exploited to break through protection. In this work, we present an alternative lightweight strategy called Paired-Point Lifting (PPL) for constructing 3D line clouds. Instead of drawing one randomly oriented line per 3D point, PPL splits 3D points into pairs and joins each pair to form 3D lines. This seemingly simple strategy yields 3 benefits, i) new ambiguity in feature selection, ii) increased line cloud sparsity and iii) non-trivial distribution of 3D lines, all of which contributes to enhanced protection against privacy attacks. Extensive experimental results demonstrate the strength of PPL in concealing scene details without compromising localization accuracy, unlocking the true potential of 3D line clouds.

## :mag:Dataset

We utilized two datasets [Learning to navigate the energy landscape](https://graphics.stanford.edu/projects/reloc/) and [Cambridge Landmarks](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3), which we denoted as **_Energy landscape_** and **_Cambridge_** in the paper.

## :running: How to run our code!

- **Environment setting**

Make a new folder `/Myfolder`.
Make a docker container that fits your environment with a python version 3.9.
Mount the docker volume with the `-v /Myfolder/:/workspace/`.

Clone the git `git clone https://github.com/Fusroda-h/ppl`

Download `eigen-3.4.0.tar.gz` library from https://eigen.tuxfamily.org/index.php?title=Main_Page to run poselib.

ex)
```bash
cd ppl
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
```

To properly build `poselib`, download the rest of the folders from the [PoseLib](https://github.com/vlarsson/PoseLib).
We only uploaded the customized code from PoseLib implementing P6L solver.

ex)
```bash
cd ..
git clone https://github.com/PoseLib/PoseLib.git
# Checkout to the version before refactoring "pybind"
cd PoseLib
git checkout 0b70bf0ae23a95e9d098d278562ebd8c6e59ae0d
# Overwrite customized local poselib to cloned poselib
# And move back to original directory
cd ../
cp -rf ppl/PoseLib/* PoseLib/
rm -r ppl/PoseLib
mv PoseLib ppl/PoseLib
```

Since InvSfM code by Pittaluga et al. is written in tensorflow.v1, Chanhyuk Yun rewritten the whole code to pytorch for the ease of use ([invsfm_torch](https://github.com/ChanhyukYun/invSfM_torch)).
Download pretrained weights from [InvSfM](https://github.com/francescopittaluga/invsfm).
Position the `wts` folder to `utils/invsfm/wts`.
Then, our code will automatically change the weights to torch version and utilize it.

```bash
cd ppl
bash start.sh
```

cf) If you suffer from an initialization error with the message: `avx512fintrin.h:198:11: note: ‘__Y’ was declared here`.

Refer to this [ISSUE](https://github.com/pytorch/pytorch/issues/77939#issue-1242584624) and install with GCC-11

`apt-get install gcc-11 g++-11`

Edit the bash file `start.sh` so that Poselib is compiled with `gcc-11` $-$ substitute `cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install`
to `cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11`.


If you have other problems in building the packages.
Visit installation each page, s.t. [PoseLib](https://github.com/vlarsson/PoseLib), [Ceres-solver](http://ceres-solver.org/installation.html), [COLMAP](https://colmap.github.io/install.html).
Ubuntu and CUDA version errors might occur.

The codes `database.py` and `read_write_model.py` is from [COLMAP](https://github.com/colmap/colmap).

- **Run the main code (pose estimation, recovering point, restoring image at once)**

You can download example dataset on [Sample_data](https://1drv.ms/u/s!AlaAkmWU9TVG6yIqNBD0PlN43Ewe?e=2gIN1F).
Directories are organized like below.
```bash
├─Dataset_type (energy, cambridge)
│  └─Scene (apt1_living, kingscolledge)
│      ├─bundle_maponly
│      ├─images_maponly
│      ├─query
│      ├─sparse_gt
│      ├─sparse_maponly
│      └─sparse_queryadded
```
The construction of map and queries are explained in [supplementary materials](documents/Lee_et_al_cvpr23_supplemat.pdf).

To generate the PPL-based line cloud and to estimate pose & recover the point cloud from this

```
/usr/local/bin/python main.py
```

You can change your options with the parser in `main.py`.
Or else can manipulate the miute options with `static/variable.py`.

The results are stored in `output` folder.
In the folder, recovered point clouds, pose errors, and recovered image qualities are stored in `Dataset_name/Scene/L2Precon`,`Dataset_name/Scene/PoseAccuracy`,`Dataset_name/Scene/Quality` respectively.
The recovered images will be saved in `dataset/Dataset_name/Scene/invsfmIMG/`.


## Citation

[Summarize your findings and discuss their implications for future research.]

```bibtex
@inproceedings{lee2023ppl,
  title={Paired-Point Lifting for Enhanced Privacy-Preserving Visual Localization},
  author={Lee, Chunghwan and Kim, Jaihoon and Yun, Chanhyuk and Hong, Je Hyeong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17266--17275},
  year={2023}
}
```

## License
A patent application for the Paired-point lifting (PPL) algorithm and the relevant software has been submitted and is under review for registration.
Paired-point lifting (PPL) is licensed under the CC-BY-NC-SA-4.0 license limiting any commercial use.
[PoseLib](https://github.com/vlarsson/PoseLib) is licensed under the BSD 3-Clause license.

## Acknowledgement
- This work was partly supported by the National Research Foundation of Korea(NRF) grants funded by the Korea government(MSIT) (No. 2022R1C1C1004907 and No. 2022R1A5A1022977), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (Artificial Intelligence Graduate School Program(Hanyang University)) (2020-0-01373) and Hanyang University (HY-202100000003084).

