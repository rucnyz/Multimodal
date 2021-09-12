#### Model

The TMC is the module used for the several multiview dataset

#### Environement

Create a 3.8.10 python environement with:

```
sklearn            0.24.2
scipy              1.6.2
torch              1.9.0    
torchvision        0.10.0 
tensorboard        2.5.0
numpy              1.20.2                 
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

- [UCI数据集来源&简要测试](https://github.com/mvlearn/mvlearn)
- [多模态数据集来源](https://github.com/yeqinglee/mvdata)

#### Related Code：

- [借鉴的代码框架](https://github.com/jbdel/MOSEI_UMONS)
- [借鉴的多模态分类模型](https://github.com/hanmenghan/TMC)
- [mimic数据预处理](https://github.com/YerevaNN/mimic3-benchmarks)

#### Training

To train a TMC model on the UCI labels, use the following command :

```
python main.py --model TMC --name mymodel --seed 123 --batch_size 200 --lr_base 0.0003 --dataset UCI
```

Checkpoints are created in folder `ckpt/mymodel`

#### Evaluation

You can evaluate a model by typing :

```
python ensembling.py --name mymodel
```

The task settings are defined in the checkpoint state dict, so the evaluation will be carried on the dataset you trained
your model on.

By default, the script globs all the training checkpoints inside the folder and ensembling will be performed.

#### Results:

UCI dataset

|训练集缺失3模态的比例|Accuracy(234模态)<br>(100epoch)|Accuracy(234模态)<br>(700epoch或达到提前结束训练条件)|Accuracy(全模态)<br>(700epoch或达到提前结束训练条件)|
|:----|:----|:----|:----|
|100%|83.25|92|96.5|
|50%|83.75|91.25(379)|96.5|
|25%|80.5|93.75|96.5|
|5%|89.25|93.75(477)|96.25|
|2.5%|91.0|94(497)|96.25|
|0%|92.25|94(520)|96.25|

#### Pre-trained checkpoints:

