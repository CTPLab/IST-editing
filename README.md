# IST-editing
Official PyTorch implementation for the manuscript accepted by **MIDL (oral)**:

**IST-editing: Infinite spatial transcriptomic editing in a generated gigapixel mouse pup**

Jiqing Wu, Ingrid Berg and Viktor H. Koelzer.

Camera-ready version: https://openreview.net/pdf?id=9pBGVsHdzL

If you find this repository helpful for your research, we would appreciate your citation of the following papers.
```
@inproceedings{wu2024ist,
  title={IST-editing: Infinite spatial transcriptomic editing in a generated gigapixel mouse pup},
  author={Wu, Jiqing and Berg, Ingrid and Koelzer, Viktor H},
  booktitle={International Conference on Medical Imaging with Deep Learning},
  year={2024},
  organization={PMLR}
}

@article{10.1093/bioinformatics/btae077,
    author = {Wu, Jiqing and Koelzer, Viktor H},
    title = {SST-editing: in silico spatial transcriptomic editing at single-cell resolution},
    journal = {Bioinformatics},
    volume = {40},
    number = {3},
    pages = {btae077},
    year = {2024},
    month = {02},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btae077},
    url = {https://doi.org/10.1093/bioinformatics/btae077},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/40/3/btae077/56850944/btae077.pdf},
}
```

<p align="center">
<img src="visual/model.png" width="1600px"/>  
<br>
Conceptual illustrations of the proposed model.       
</p>


<p align="center">
<img src="visual/generated_WSI.png" width="1600px"/>  
<br>
Overview for the generated WSI of the mouse pup.       
</p>


## Prerequisites
The code base has been successfully tested under the following configurations:

- Ubuntu 20.04
- Nvidia driver 525.147
- CUDA 11.8
- Python 3.11
- PyTorch 2.1
- Miniconda 

Note that lower PyTorch version, E.g., >= 1.13 should also work.

Please check [environment.yml](environment.yml) for creating the conda environment.
In case pyvips package does not work properly, install it via conda commmand instead,
```
conda install --channel conda-forge pyvips
```

## Data Preparation
- Unzip the downloaded [mouse.zip](https://zenodo.org/records/10404337/files/mouse.zip?download=1)  to Data/Xenium_mouse folder (create it if not exists). 


If you want to process the dataset from scratch, you could first download the raw data via 

https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.6.0/Xenium_V1_mouse_pup/Xenium_V1_mouse_pup_outs.zip

Please see also [prep_inf.py](Dataset_inf/prep_inf.py) and [utils_mouse.py](Dataset_inf/utils_mouse.py) for more implementation details about the data processing pipeline.

## Train the IST-editing model 
Once the mouse dataset is ready, run the [train.sh](train.sh) script for training the model on a single GPU (e.g., RTX 2080/3090 Ti)

```
sh train.sh
```

In case of multi-gpu training (let's say 2), replace the first line in [train.sh](train.sh) with 

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4096 train_style2_inf.py \
```

## Generate the WSI of the mouse pup
After complete the training, run the following command to generate the cell-, tissue-and animal-level images including the gigapixel WSI (saved under Path/To/MERG_0_D/\*_1_53_0_26_*.tif)  

```
TASK=all_organs_video-bbox-tif_tile  N_ITER=800000 sh infer.sh
```

## Run the gene expression-guided editing 
For instance, run the following command to conduct direct scaling of gene expressions w.r.t. selected tissue regions

```
TASK=all_organs_video-scale_tile  N_ITER=800000 sh infer.sh
```

## The gigapixel generated WSIs 
The downloading links to raw and generated WSIs, which can be examined by open-sourced QuPath or commericial software such as HALO

- [Raw DAPI](https://zenodo.org/records/10404337/files/dapi_raw.tif?download=1)
- [Generated DAPI](https://zenodo.org/records/10404337/files/dapi_gen.tif?download=1)
- [Raw H&E](https://zenodo.org/records/10404337/files/he_raw.tif?download=1)
- [Generated H&E](https://zenodo.org/records/10404337/files/he_gen.tif?download=1)


## Supplementary generalization experiments
- Unzip the downloaded [brain.zip](https://zenodo.org/records/10816781/files/brain_60988.zip?download=1)  to Data/MERFISH folder (create it if not exists). 

- To train the model, run
  ```
  sh train_brn.sh
  ```

- To generate the cell-, tissue-and animal-level images including the gigapixel WSI 

  ```
  TASK=all_organs_video-bbox-tif_tile  N_ITER=800000 sh infer_brn.sh
  ```
- You could examine the generated 73728 X 106496 WSIs for training and test via https://zenodo.org/records/10816781.


## Acknowledgment
This repository is developed upon [SST-editing](https://github.com/CTPLab/SST-editing) project. 
Besides, we would like to thank all the researchers contributing to the Xenium mouse pup dataset.

## License
The copyright license of this repository is specified with the LICENSE-IST-editing.
