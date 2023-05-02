# Adversarial Perturbation Elimination with GAN(APE-GAN)

Reference paperfor defense (https://arxiv.org/pdf/1707.05474.pdf)

## MNISTdataset

### 1. Train CNN(Victim model) and Generate Adversarial Examples(FGSM)
```
python generate.py --eps 0.15
```

### 2. Train Adversarial Perturbation Elimination with GAN
```
python train.py --checkpoint ./checkpoint/mnist
```

### 3. Test
```
python test_model.py --eps 0.15 --gan_path ./checkpoint/mnist/3.tar
```

