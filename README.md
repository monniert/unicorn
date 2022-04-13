# UNICORN :unicorn:

<h3>
<a class="label label-info" href="http://imagine.enpc.fr/~monniert/UNICORN/">Webpage</a> |
<a href="https://arxiv.org/abs/todo">Paper</a> |
<a href="https://arxiv.org/abs/todo">Supplementary</a> |
<a href="http://imagine.enpc.fr/~monniert/UNICORN/ref.bib">BibTex</a>
</h3>

![car.gif](./media/car.gif)
![bird.gif](./media/bird.gif)
![moto.gif](./media/moto.gif)

Official PyTorch implementation of [**"Share With Thy Neighbors: Single-View Reconstruction 
by Cross-Instance Consistency"**](https://arxiv.org/abs/todo) paper, check out our 
[**webpage**](https://imagine.enpc.fr/~monniert/UNICORN) for details!

If you find this code useful, don't forget to star the repo :star: and cite the paper:

```
@article{monnier2022unicorn,
  title={{Share With Thy Neighbors: Single-View Reconstruction by Cross-Instance 
  Consistency}},
  author={Monnier, Tom and Fisher, Matthew and Efros, Alexei A and Aubry, Mathieu},
  journal={arXiv:todo},
  year={2022},
}
```

## Installation :construction_worker:

### 1. Create conda environment :wrench:

```
conda env create -f environment.yml
conda activate unicorn
```

**Optional:** some monitoring routines are implemented, you can use them by specifying your
visdom port in the config file. You will need to install `visdom` from source beforehand

```
git clone https://github.com/facebookresearch/visdom
cd visdom && pip install -e .
```

### 2. Download datasets :arrow_down:

```
bash scripts/download_data.sh
```

This command will download one of the following datasets:

- `ShapeNet NMR`: [paper](https://arxiv.org/abs/1512.03012) / [NMR 
  paper](https://arxiv.org/abs/1711.07566) / 
  [dataset](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip)
  (33Go, thanks to the [DVR 
  team](https://github.com/autonomousvision/differentiable_volumetric_rendering) for hosting 
  the data)
- `CUB-200`: [paper](https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf) / 
  [webpage](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) /
  [dataset ](https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/) 
  (1Go)
- `Pascal3D+ Cars`: [paper](https://cvgl.stanford.edu/papers/xiang_wacv14.pdf) /
  [webpage](https://cvgl.stanford.edu/projects/pascal3d.html) (including ftp download link, 
  7.5Go) / [UCMR
  annotations](https://people.eecs.berkeley.edu/~shubham-goel/projects/ucmr/cachedir-others.tar.gz)
  (thanks to the [UCMR team](https://github.com/shubham-goel/ucmr/) for releasing them)
- `CompCars`: [paper](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/CompCars.pdf) / 
  [webpage](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) / 
  [dataset](https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/comprehensive_cars.zip) 
  (12Go, thanks to the [GIRAFFE team](https://github.com/autonomousvision/giraffe/) for 
  hosting the data)
- `LSUN`: [paper](http://arxiv.org/abs/1506.03365) / [webpage](https://www.yf.io/p/lsun) / 
  [horse dataset](http://dl.yf.io/lsun/objects/horse.zip) (69Go) / [moto 
  dataset](http://dl.yf.io/lsun/objects/motorbike.zip) (42Go)

### 3. Download pretrained models :arrow_down:

```
bash scripts/download_model.sh
```

This command will download one of the following models:

- `car.pkl` trained on CompCars: [gdrive 
  link](https://drive.google.com/file/d/16aIw88ZiAUFUOOBFXdHOUNtJ1-w3zpJG/view?usp=sharing)
- `bird.pkl` trained on CUB-200: [gdrive 
  link](https://drive.google.com/file/d/1nWrmMCjeJzK5nHhZ021CCYS-51LTpKHe/view?usp=sharing)
- `moto.pkl` trained on LSUN Motorbike: [gdrive 
  link](https://drive.google.com/file/d/1wuVjllVUSVWUyfoleSHd2qKiET-x-l1i/view?usp=sharing)
- `horse.pkl` trained on LSUN Horse: [gdrive 
  link](https://drive.google.com/file/d/1DoJ0HQ60veEPTmWB4JJ_NGQa5U_48Yhs/view?usp=sharing)
- `sn_*.pkl` trained on each ShapeNet category: 
  [airplane](https://drive.google.com/file/d/1WkqfL7zoOrPegHoZCFxy8kTI_DTnFj1W/view?usp=sharing), 
  [bench](https://drive.google.com/file/d/1__EgJZTtz2y3xI963vgY6j3-kTz8tHaC/view?usp=sharing), 
  [cabinet](https://drive.google.com/file/d/1Yql_enYUniDDP8HXhQ-ZD9hWuvpI6wj6/view?usp=sharing), 
  [car](https://drive.google.com/file/d/1nF_xJfdUsepUkN-i88WaJYpR8RHKCCxI/view?usp=sharing),
  [chair](https://drive.google.com/file/d/1sDdERppgW-q3pCoATCcbVrBNsY5cbPB6/view?usp=sharing),
  [display](https://drive.google.com/file/d/1q93zt9cJKO4rrNkQ2NkqqIHikd2xm0LG/view?usp=sharing),
  [lamp](https://drive.google.com/file/d/1kDV9ulT9ip1cQKamauX-YgPFuCnF1w3G/view?usp=sharing),
  [phone](https://drive.google.com/file/d/1MpUnyb9w6ZE7_EKUkz35JdADRueW9zDO/view?usp=sharing),
  [rifle](https://drive.google.com/file/d/1L5TXJldoeoBshgHuPd3rsSAmz_lCMnjt/view?usp=sharing),
  [sofa](https://drive.google.com/file/d/1u2Mi4hf2_pfmWVLEcsrekaNcrK-6XOew/view?usp=sharing),
  [speaker](https://drive.google.com/file/d/1ZoEOmtnB6aYH05fD0tJba038Wbk1ZLf7/view?usp=sharing),
  [table](https://drive.google.com/file/d/1MwGZpFaadA-3fA1WpXKmX-v7btXcuZJ7/view?usp=sharing),
  [vessel](https://drive.google.com/file/d/1-2Jwek4GmYDciRNu2K6zsMlyW7c3krBl/view?usp=sharing)

**NB:** it may happen that `gdown` hangs, if so you can download them manually with following 
gdrive links and move them to the `models` folder.


## How to use :rocket:

### 1. 3D reconstruction of car images :oncoming_automobile:

![ex_car.png](./media/ex_car.png)
![ex_rec.gif](./media/ex_rec.gif)

You first need to download the car model (see above), then launch:

```
cuda=gpu_id model=car.pkl input=demo ./scripts/reconstruct.sh
```

where:
- `gpu_id` is a target cuda device id,
- `car.pkl` corresponds to a pretrained model,
- `demo` is a folder containing the target images.

It will create a folder `demo_rec` containing the reconstructed meshes (.obj format + gif 
visualizations).

### 2. Reproduce our results :bar_chart:

![shapenet.gif](./media/shapenet.gif)

To launch a training from scratch, run:

```
cuda=gpu_id config=filename.yml tag=run_tag ./scripts/pipeline.sh
```

where:
- `gpu_id` is a target cuda device id,
- `filename.yml` is a YAML config located in `configs` folder,
- `run_tag` is a tag for the experiment.

Results are saved at `runs/${DATASET}/${DATE}_${run_tag}` where `DATASET` is the dataset name 
specified in `filename.yml` and `DATE` is the current date in `mmdd` format. Some training 
visual results like reconstruction examples will be saved. Available configs are:

- `sn/*.yml` for each ShapeNet category
- `car.yml` for CompCars dataset
- `cub.yml` for CUB-200 dataset
- `horse.yml` for LSUN Horse dataset
- `moto.yml` for LSUN Motorbike dataset
- `p3d_car.yml` for Pascal3D+ Car dataset

### 3. Train on a custom dataset :crystal_ball:

If you want to learn a model for a custom object category, here are the key things you need 
to do:

1. put your images in a `custom_name` folder inside the `datasets` folder
2. write a config `custom.yml` with `custom_name` as `dataset.name` and move it to the `configs` folder: as a rule of thumb for the progressive conditioning milestones, put the number of epochs corresponding to 500k iterations for each stage
3. launch training with:

```
cuda=gpu_id config=custom.yml tag=custom_run_tag ./scripts/pipeline.sh
```

## Further information :books:

If you like this project, check out related works from our group:

- [Monnier et al. - Unsupervised Layered Image Decomposition into Object Prototypes (ICCV
  2021)](https://arxiv.org/abs/2104.14575)
- [Monnier et al. - Deep Transformation Invariant Clustering (NeurIPS 
  2020)](https://arxiv.org/abs/2006.11132)
- [Deprelle et al. - Learning elementary structures for 3D shape generation and matching 
  (NeurIPS 2019)](https://arxiv.org/abs/1908.04725)
- [Groueix et al. - AtlasNet: A Papier-Mache Approach to Learning 3D Surface Generation (CVPR 
  2018)](https://arxiv.org/abs/1802.05384)
