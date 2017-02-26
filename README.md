# tensorflow SVHN example
This is a tensorflow example to train and test the SVHN Dataset (Street View House Numbers Dataset http://ufldl.stanford.edu/housenumbers/).

# Here are some notes.
 * train only on train_32x32.mat (without extra_32x32.mat)
 * test on test_32x32.mat
 * achieve 0.81 acc by code/nn/
 * achieve 0.93 acc by code/cnn/
 * running well on tensorflow gpu version 0.12

# Run NN on SVHN
The NN architecture is simple two-layer perceptron
```bash
cd code/
python download_svhn.py
cd nn/
python train.py && python test.py
```

# Run CNN on SVHN
The CNN architecture is based from VGGNet
```bash
cd code/
python download_svhn.py
cd cnn/
python train.py && python test.py
```