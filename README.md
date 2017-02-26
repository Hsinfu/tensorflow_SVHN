# tensorflow_SVHN_example
This is a simple example to train and test the Street View House Numbers (SVHN) Dataset (http://ufldl.stanford.edu/housenumbers/).

# Here are some notes.
 * train only on train_32x32.mat (without extra_32x32.mat)
 * test on test_32x32.mat
 * achieve 0.81 acc by code/nn/
 * achieve 0.93 acc by code/cnn/
 * running well on tensorflow gpu version 0.12

# Run NN on SVHN
```bash
cd code/
python download_svhn.py
cd nn/
python train.py && python test.py
```

# Run CNN on SVHN
```bash
cd code/
python download_svhn.py
cd cnn/
python train.py && python test.py
```