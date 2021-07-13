#### Model

The Model_LA is the module used for the UMONS solution to the MOSEI dataset using only linguistic and acoustic
inputs.

#### Environement

Create a 3.8.5 python environement with:

```
sklearn            0.24.2
spacy              2.3.7
torch              1.9.0    
torchvision        0.10.0 
tensorboard        2.5.0
numpy              1.20.2    
```

We use GloVe vectors from space by using the following codes :

```
import spacy
spacy_tool = spacy.load("en_vectors_web_lg")
```
Change the default version of python in macOS
```
打开终端
brew update
brew install pyenv
echo 'eval "$(pyenv init --path)"' >> ~/.zprofile

pyenv versions(如果出现command not found重新打开terminal)
pyenv global 3.x(上一步中列出的3版本的那一个)
(再重新打开应该就生效了)
```
#### Data

Download data from [here](https://drive.google.com/file/d/1tcVYIMcZdlDzGuJvnMtbMchKIK9ulW1P/view?usp=sharing).
<br/>Unzip the files into the 'data' folder<br/>
More information about the data can be found in the 'data' folder<br/>

#### Training

To train a Model_AV model on the emotion labels, use the following command :

```
python main.py --model Model_LA --name mymodel --task emotion --multi_head 4 --ff_size 1024 --hidden_size  512 --layer 4 --batch_size 32 --lr_base 0.0001 --dropout_r 0.1
```

Checkpoints are created in folder `ckpt/mymodel`

Argument `task` can be set to `emotion` or `sentiment`. To make a binarized sentiment training (positive or negative),
use `--task_binary True`

#### Evaluation

You can evaluate a model by typing :

```
python ensembling.py --name mymodel
```

The task settings are defined in the checkpoint state dict, so the evaluation will be carried on the dataset you trained
your model on.

By default, the script globs all the training checkpoints inside the folder and ensembling will be performed.

#### Results:

Results are run on a single GeForce GTX 1080 Ti.<br>
Training performances:
| Modality | Memory Usage | GPU Usage | sec / epoch | Parameters | Checkpoint size | | ------------- |:-------------:|:
-------------:|:-------------:|:-------------:|:-------------:| | Linguistic + acoustic | 320 Mb | 2400 MiB | 103 | ~ 33
M | 397 Mb | Linguistic + acoustic + vision |

You should approximate the following results :

| Task Accuracy  |     val | test | test ensemble | epochs | 
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| Sentiment-7    | 43.61   |  43.90  | 45.36  | 6
| Sentiment-2    |  82.30  |  81.53  | 82.26  |  8
| Emotion-6      | 81.21   |  81.29  | 81.48  |  3

Ensemble results are of max 5 single models <br>
7-class and 2-class sentiment and emotion models have been train according to the
instructions [here](https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/README.md).<br>

#### Pre-trained checkpoints:

Result `Sentiment-7 ensemble` is obtained from these
checkpoints : [Download Link](https://drive.google.com/file/d/11BKBbxp2tNZ6Ai1YD-pPrievffYh7orM/view?usp=sharing)<br/>
Result `Sentiment-2 ensemble` is obtained from these
checkpoints : [Download Link](https://drive.google.com/file/d/15PanBXsxXzvmDsVuA5qiWQd33ssezjxn/view?usp=sharing)<br/>
Result `Emotion ensemble` is obtained from these
checkpoints : [Download Link](https://drive.google.com/file/d/1GyXRWhtf0_sJQacy5wT8vHoynwHkMo79/view?usp=sharing)<br/>
