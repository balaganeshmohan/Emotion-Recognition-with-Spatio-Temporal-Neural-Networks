

<h3  align="center">Emotion Recognition with Spatio-Temporal Neural Networks</h3>

  

<p  align="center">



</p>

</p>

  
 
  


  
<!-- ABOUT THE PROJECT -->

## About The Project
*This project is part of my University Thesis. This repo contains code to train neural network models to recognize emotions with facial expressions. In particular, two different architectures were investigated, one of which is a CNN-based model called Temporal Shift Module (TSM) that can learn spatio-temporal features in 3D data with the computational complexity of a 2D CNN and the other one, which is a video based vision transformer, employing spatio-temporal self attention. The models were trained and tested on the CREMA-D dataset, as well as on the In the Wild v2 dataset for testing the generalization capabilities of the aforementioned methods, achieving state-of-the-art performance.*
  



  



  

### Built With
This project utilizes code from mmaction2 and Slowfast based implementations of TSM and TimeSformer networks. 
  
* [Python](https://www.python.org/)

* [MMaction2](https://github.com/open-mmlab/mmaction2)

* [TimeSformer](https://github.com/facebookresearch/TimeSformer)

  
  
  

<!-- GETTING STARTED -->

## Getting Started

  Download the datasets from [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) and [In the Wild v2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)

  

### Prerequisites

  



* python
* PyTorch
* mmaction 2
* Slowfast




 
### Installation

  

1. Clone the repo

2. Go to folder, and do a pip install.

```sh

Emotion recognition/TSM/requirements.txt 

```

3. Run both the setup files.

```sh

Emotion recognition/TSM/setup.py
Emotion recognition/TimeSformer/setup.py

```

  
  
  

<!-- USAGE EXAMPLES -->

## Usage
### Data preparation

  Prepare the Data with scripts provided.  AffWild needs to be  split with the script split_main. Do a facial extraction with script FaceNet if necessary, do a key frame extraction with the last script.
  ```sh
Emotion recognition/prep_data/split_videos/split_main.py
Emotion recognition/prep_data/face_detector/FaceNet.py
Emotion recognition/prep_data/key_frame/main.py
```
Create a list with the information in this format,
```sh
Video_path Label
FEA/1044_DFA_FEA_XX/1044_DFA_FEA_XX.mp4 3
NEU/1058_ITS_NEU_XX/1058_ITS_NEU_XX.mp4 4
```
Now optionally, you can run ***train_val_test.ipynb*** to split the label list into three segments for training, testing, and validation.

### Train
To train TSM, edit the config file. you can experiment with different data and training/validation/testing parameters. Mainly change the data and labels directory. 
```sh
TSM/configs/tsm_r50_video_1x1x8_train.py
```
Initiate training with 
```sh
python TSM/tools/train.py --path_to_config
```

Similarly for TimeSformer, change the directories in the config file.
```sh
TimeSformer/configs/TimeSformer_divST_8x32_224.yaml
```
Initiate training with 
```sh
python TimeSformer/tools/run_net.py --path_to_config
```
  
  ### Test
Testing is very similar. Edit the config file to change Train as False. 
```sh
TSM
python TSM/tools/test.py --path_to_config --path_to_ckpt --out_dir

TimeSformer
python TimeSformer/tools/run_net.py --path_to_config --path_to_ckpt --out_dir


```
### Demo
Alternatively, you can download the weights from the checkpoint files, and run a demo file to predict results on your own data with [TSM AffWild](https://drive.google.com/file/d/1osK2L8q8DLbWiQRNY6ne-I3lxG8BaIZ9/view?usp=sharing), [TSM CREMA-D](https://drive.google.com/file/d/1_CssBgsSHkOeSQ37aONRiUV-c2kAXwU-/view?usp=sharing).

  ```sh
  TSM/demo/demo.py -path_to_config --ckpt_file --label_file 
					--out_dir_file --fps --resolution
```
    

<!-- LICENSE -->

## License

  

Distributed under the MIT License. 

  
  
  



  
  


