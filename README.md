
# AGSD-Surgical-Instrument-Segmentation

Code for 'Unsupervised Surgical Instrument Segmentation via Anchor Generation and Semantic Diffusion' (MICCAI 2020).

[Paper](https://arxiv.org/abs/2008.11946) and [Video Demo](http://www.vie.group/media/pdf/demo.mp4).

![ ](https://github.com/Finspire13/AGSD-Surgical-Instrument-Segmentation/blob/master/plot.png)

## Setup
* Recommended Environment: Python 3.5, Cuda 10.0, PyTorch 1.3.1
* Install dependencies: `pip3 install -r requirements.txt`.

## Data
 1. Download our data for EndoVis 2017 from [Baidu Yun](https://pan.baidu.com/s/1qDq38oiO7DunwVYYNQ_dSQ) (PIN:m0o7) or [Google Drive]()(TO DO).
 2. Unzip the file and put into the current directory.
 3. The data includes following sub-directories:

`image`  : Raw images (Left frames) from the [EndoVis 2017 dataset](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/) 

`ground_truth`  : Ground truth of binary surgical instrument segmentation.

`cues`  : Hand-designed coarse cues for surgical instruments.

`anchors`  : Anchors generated by fusing cues.

`prediction`  : Final probability maps output by our trained model (Single stage setting).

## Run

Simply run `python3 main.py --config config-endovis17-SS-full.json` .

This config file `config-endovis17-SS-full.json` is for the full model in the single stage setting (SS).

For other experimental settings in our paper, please accordingly modify the config file and the `train_train_datadict`, `train_test_datadict`, `test_datadict` in `main.py` if necessary.

## Output

Results will be saved in a folder named with the `naming` in the config file. 

This output folder will include following sub-directories:

`logs` : A Tensorboard logging file and an numpy logging file.

`models`: Trained models.

`pos_prob`: Probability maps for instruments.

`pos_mask`: Segmentation masks for instruments.

`neg_prob`: Probability maps for non-instruments.

`neg_mask`: Segmentation masks for non-instruments.


## Citation
TO DO.

## License
MIT

