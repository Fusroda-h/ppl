# PPL (Paired-point lifting)
## Paired-Point Lifting for Enhanced Privacy-Preserving Visual Localization (CVPR 2023)

**Authors:** Chunghwan Lee, Jaihoon Kim, Chanhyuk Yun, [Je Hyeong Hong](https://sites.google.com/view/hyvision)

![Teaser](/path/to/image.jpg)

https://github.com/Fusroda-h/ppl/assets/64474003/78b7ecb5-717e-44a6-96fb-aed6cb8a4a59


https://github.com/Fusroda-h/ppl/assets/64474003/167f1f00-7989-45a5-9f2f-896387ccde82


https://github.com/Fusroda-h/ppl/assets/64474003/aad1fc65-5f88-4930-a73a-2234cf09e7fc


**Abstract:** Visual localization refers to the process of recovering camera pose from input image relative to a known scene, forming a cornerstone of numerous vision and robotics systems. While many algorithms utilize sparse 3D point cloud of the scene obtained via structure-from-motion (SfM) for localization, recent studies have raised privacy concerns by successfully revealing high-fidelity appearance of the scene from such sparse 3D representation. One prominent approach for bypassing this attack was to lift 3D points to randomly oriented 3D lines thereby hiding scene geometry, but latest work have shown such random line cloud has a critical statistical flaw that can be exploited to break through protection. In this work, we present an alternative lightweight strategy called Paired-Point Lifting (PPL) for constructing 3D line clouds. Instead of drawing one randomly oriented line per 3D point, PPL splits 3D points into pairs and joins each pair to form 3D lines. This seemingly simple strategy yields 3 benefits, i) new ambiguity in feature selection, ii) increased line cloud sparsity and iii) non-trivial distribution of 3D lines, all of which contributes to enhanced protection against privacy attacks. Extensive experimental results demonstrate the strength of PPL in concealing scene details without compromising localization accuracy, unlocking the true potential of 3D line clouds.

## :mag:Dataset

[Provide a brief introduction to your research, including the background and context of your study, and explain why your research is important.]

## How to run

![Fancy Figure](/path/to/image.jpg)

Download `eigen-3.4.0.tar.gz` library from https://eigen.tuxfamily.org/index.php?title=Main_Page to run poselib.
Make a docker container that fits your environment with a python version 3.9.

To properly build `poselib`, download the rest of the folders from the [PoseLib]{https://github.com/vlarsson/PoseLib}.
We only uploaded the customized code from PoseLib implementing P6L solver.

Since InvSfM code by Pittaluga et al. is written in tensorflow.v1, Chanhyuk Yun rewritten the whole code to pytorch for the ease of use ([invsfm_torch]{https://github.com/ChanhyukYun/invSfM_torch}).
Download pretrained weights from [InvSfM]{https://github.com/francescopittaluga/invsfm}.
Position the `wts` folder to `utils/invsfm/wts`.
Then, our code will automatically change the weights to torch version and utilize it.

```bash
bash start.sh
```
The codes `database.py` and `read_write_model.py` is from [COLMAP]{https://github.com/colmap/colmap}.



## Results

[Describe your findings in detail, including any graphs or figures that help illustrate your results.]

![Fancy Figure](/path/to/image.jpg)

## Citation

[Summarize your findings and discuss their implications for future research.]

```bibtex
@inproceedings{lee2023paired,
  title={Paired-Point Lifting for Enhanced Privacy-Preserving Visual Localization},
  author={Lee, Chunghwan and Kim, Jaihoon and Yun, Chanhyuk and Hong, Je Hyeong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17266--17275},
  year={2023}
}
```
## License
Paired-point lifting (PPL) algorithm has been patented and is in the process of registration.
Paired-point lifting (PPL) is licensed under the CC0-1.0 license.
[PoseLib](https://github.com/vlarsson/PoseLib) is licensed under the BSD 3-Clause license.

## Acknowledgement

[List any references you cited in your paper here.]
- This work was partly supported by the National Research Foundation of Korea(NRF) grants funded by the Korea government(MSIT), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (Artificial Intelligence Graduate School Program(Hanyang University)) and Hanyang University.

