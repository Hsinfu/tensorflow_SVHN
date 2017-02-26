from os.path import isfile, join
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

dataset_dir = 'data/'
dataset_url = 'http://ufldl.stanford.edu/housenumbers/'
dataset = ['train_32x32.mat', 'test_32x32.mat']
# dataset = ['train_32x32.mat', 'test_32x32.mat', 'extra_32x32.mat']

for data in dataset:
    path = join(dataset_dir, data)
    url = join(dataset_url, data)
    if not isfile(path):
        print('downloading %s' % data)
        urlretrieve(url, path)
