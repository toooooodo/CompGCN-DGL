## Composition-Based Multi-Relational Graph Convolutional Networks

An implement of **CompGCN** in [Pytorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/).
- Paper: [ICLR 2020: Composition-Based Multi-Relational Graph Convolutional Networks](https://openreview.net/forum?id=BylA_C4tPr)
- Author's code: [https://github.com/malllabiisc/CompGCN](https://github.com/malllabiisc/CompGCN)

### Dependencies
1. install Python3
2. install requirements `pip install -r requirements.txt`

### Train Model
To start training process:

```shell script
python run.py --score_func conve --opn corr --gpu 0 --epoch 500 --batch 256 --n_layer 1
```

  - `--score_func` denotes the link prediction score score function 
    - `conve` : [Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/abs/1707.01476)
    - `distmult` : [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575)
  - `--opn` is the composition operation used in **CompGCN**. It can take the following values:
    - `sub` for subtraction operation:  Φ(e_s, e_r) = e_s - e_r
    - `mult` for multiplication operation:  Φ(e_s, e_r) = e_s * e_r
    - `corr` for circular-correlation: Φ(e_s, e_r) = e_s ★ e_r
  - `--gpu` for specifying the GPU to use
  - `--epoch` for number of epochs
  - `--batch` for batch size
  - `--n_layer` for number of GCN Layers to use
  - Rest of the arguments can be listed using `python run.py -h`

### Test Result
#### FB15k-237

model | MRR | MR | Hits@1 | Hits@3 | Hits@10
:-: | :-: | :-: | :-: | :-: | :-: 
Conve-mult | 0.35516 | 222 | 0.26337 | 0.39021 | 0.53767
Conve-corr | 0.35340 | 199 | 0.26146 | 0.38778 | 0.53791
DistMult-mult | 0.33590 | 234 | 0.24697 | 0.36656 | 0.51639
DistMult-corr | 0.33680 | 242 | 0.24875 | 0.36624 | 0.51575

