### Results

- AMII
    - 有监督
        - 0 3090 73w68qpj 66.13
          {
          "lr_base": 0.0009879644320775532
          }
        - 0.1 卓越 2pkcql0s 65.24
          {
          "loop_times": 1,
          "lr_base": 0.000352350210418619,
          "GAN_start": 10,
          "dropout": 0.1
          }
        - 0.2 卓越 b1es58lj 63.81
          {
          "loop_times": 2,
          "lr_base": 0.002240496041756799,
          "GAN_start": 20,
          "dropout": 0.1
          }
        - 0.3 卓越 hys89bqf 62.92
          {
          "loop_times": 3,
          "lr_base": 0.0008919940841977162,
          "GAN_start": 50,
          "dropout": 0.1
          }
        - 0.4 3090 mbglau2w 62.39
          {
          "loop_times": 4,
          "lr_base": 0.00012388541185015973,
          "GAN_start": 40,
          "dropout": 0.4
          }
        - 0.5 3090 1rb8eiwh 61.14
          {
          "loop_times": 5,
          "lr_base": 0.00011794996836865433,
          "GAN_start": 100,
          "dropout": 0.2
          }
        - ALL 3090 4txr8qnc 61.69 61.80
          {
          "loop_times": 3,
          "lr_base": 0.00010130459758991694,
          "GAN_start": 60,
          "dropout": 0.3
          } {
          "loop_times": 3,
          "lr_base": 0.00011651539693675672,
          "GAN_start": 40,
          "dropout": 0.3
          }
    - 无监督
      - 
- AMII attention
    - 有监督
        - 0 3090 e5074c9w 68.27
          {
          "lr_base": 0.0030464770912076717
          }
        - 0.1 卓越 kw1ihq59 66.84
          {
          "loop_times": 5,
          "lr_base": 0.0021969454112271422,
          "GAN_start": 60,
          "dropout": 0.1
          }
          GAN_start > 0 65.42 65.78 65.95 66.49 66.31 66.84
          GAN_start =-1 65.06 64.71 64.88 64.71 64.88 64.71
        - 0.2 3090 r1ik0wnq 64.35
          {
          "loop_times": 3,
          "lr_base": 0.00800660671619617,
          "GAN_start": 100,
          "dropout": 0.2
          }
          GAN_start > 0 62.57 63.46 63.99 64.35 63.99 63.46
          GAN_start =-1 62.88 62.24 63.78 63.24 63.60 63.35
        - 0.3 卓越 r71fpu4c 63.96
          {
          "loop_times": 4,
          "lr_base": 0.0061669097205005775,
          "GAN_start": 100,
          "dropout": 0.1
          }
          GAN_start > 0 62.39 63.46 63.46 62.41 63.96 62.72
          GAN_start =-1 63.46 63.46 63.46 62.92 62.92 62.92
        - 0.4 3090 w0biuyhs 63.64
          {
          "loop_times": 1,
          "lr_base": 0.0059400398709745636,
          "GAN_start": 30,
          "dropout": 0.1
          }
          GAN_start > 0 61.14 63.64 62.57 62.03 61.85 61.68
          GAN_start =-1 62.39 63.10 62.57 62.57 62.03 62.39
        - 0.5 3090 csg8yu07 61.50
          {
          "loop_times": 5,
          "lr_base": 0.001700797633269079,
          "GAN_start": 60,
          "dropout": 0.1
          }
          GAN_start > 0 61.14 60.43 61.14 61.00 61.32 61.50
          GAN_start =-1 60.21 60.78 60.07 60.43 59.89 60.07
        - ALL 卓越 sjtv90r5 61.73
          {
          "loop_times": 4,
          "lr_base": 0.0012426136415178424,
          "GAN_start": 100,
          "dropout": 0.1
          }
    - 无监督
      - 

### Environement

- Create a 3.8.10 python environement with:

```
sklearn            0.24.2
scipy              1.6.2
torch              1.9.0    
torchvision        0.10.0 
tensorboard        2.5.0
numpy              1.20.2                 
```

- Change the default version of python in macOS

```
打开终端
brew update
brew install pyenv
echo 'eval "$(pyenv init --path)"' >> ~/.zprofile

pyenv versions(如果出现command not found重新打开terminal)
pyenv global 3.x(上一步中列出的3版本的那一个)
(再重新打开应该就生效了)
```

### Dataset

Dataset can be obtained [here](https://drive.google.com/drive/folders/1CXH_KYHDmwo0DHUZNaxWSGxWNfXdUNpB?usp=sharing).
put them in folder `data/`

### Related Data

- [UCI数据集来源&简要测试](https://github.com/mvlearn/mvlearn)
- [多模态数据集来源](https://github.com/yeqinglee/mvdata)

### Related Code：

- [借鉴的代码框架](https://github.com/jbdel/MOSEI_UMONS)
- [借鉴的多模态分类模型](https://github.com/hanmenghan/TMC)
- [借鉴的模态缺失模型](https://github.com/hanmenghan/CPM_Nets)
- [mimic数据预处理](https://github.com/YerevaNN/mimic3-benchmarks)

### Training

To train a CPM_GAN model on the UCI labels, use the following command :

```
--model CPM_GAN --name mymodel --seed 123 --batch_size 200 --lr_base 0.0005 --dataset UCI --num_worker 0 --missing_rate 0.3 --loop_times 4
```

You can adjust some hyperparameters in the following ways(e.g. modify the missing rate)

```
--model CPM --name mymodel --seed 123 --batch_size 200 --lr_base 0.0005 --dataset UCI --num_worker 0 --missing_rate 0.3 --loop_times 4
```

Checkpoints are created in folder `ckpt/mymodel`

### Evaluation(unfinished)

You can evaluate a model by typing :

```
python ensembling.py --name mymodel
```

The task settings are defined in the checkpoint state dict, so the evaluation will be carried on the dataset you trained
your model on.

By default, the script globs all the training checkpoints inside the folder and ensembling will be performed.

### Results:

UCI dataset

| 训练集缺失3模态的比例 | Accuracy(234模态)<br>(100epoch) | Accuracy(234模态)<br>(700epoch或达到提前结束训练条件) | Accuracy(全模态)<br>(700epoch或达到提前结束训练条件) |
|:------------|:------------------------------|:-----------------------------------------|:---------------------------------------|
| 100%        | 83.25                         | 92                                       | 96.5                                   |
| 50%         | 83.75                         | 91.25(379)                               | 96.5                                   |
| 25%         | 80.5                          | 93.75                                    | 96.5                                   |
| 5%          | 89.25                         | 93.75(477)                               | 96.25                                  |
| 2.5%        | 91.0                          | 94(497)                                  | 96.25                                  |
| 0%          | 92.25                         | 94(520)                                  | 96.25                                  |

### Pre-trained checkpoints:

