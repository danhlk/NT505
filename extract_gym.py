from gym_malware.envs.utils.pefeatures2 import PEFeatureExtractor2
import os
import numpy
from tqdm import tqdm

path = '/home/containernet/KLTN/Pesidious/independent/benign/train'
feature_extractore2 = PEFeatureExtractor2()

class FileRetrievalFailure(Exception):
    pass

def fetch_file(sha256: str) -> bytes:
    location = os.path.join(path, sha256)
    try:
        with open(location, 'rb') as infile:
            bytez = infile.read()
    except IOError:
        raise FileRetrievalFailure(
            "Unable to read sha256 from {}".format(location))

    return bytez
benign_train_data = []

for filename in tqdm(os.listdir(path), desc = "Process: ", ascii='#'):
    try:
        bytez = fetch_file(filename)
        features = feature_extractore2.extract(bytez)
        benign_train_data.append(numpy.array(features))
        del bytez, features
    except IOError:
        raise FileNotFoundError(
            "unable to extract file from {}".format(os.path.join(path, filename))
        )

numpy.savetxt(os.path.join(path, "../../../CSV/bengin_train_data.csv"), benign_train_data, delimiter=',')
del benign_train_data

# my_data = numpy.genfromtxt('foo.csv', delimiter=',')
print ('ALL DONE!')
