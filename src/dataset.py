from torch.utils.data import Dataset
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import os
from scipy import signal
import pywt
from keras.utils import to_categorical



class EEGDataset(Dataset):
    def __init__(self, data_path, subject=None, mode='train',transform=None):
        if mode=='train':
            self.X = np.load(os.path.join(data_path, "X_train_valid.npy"))
            self.y = np.load(os.path.join(data_path, "y_train_valid.npy"))
            self.person = np.load(os.path.join(data_path, "person_train_valid.npy"))
            self.y-=769
            self.X = self.X[:,:,0:500]  # trim to 500 timesteps

            # Uncomment the below lines for preprocessing:
            ## Trim --> MaxPool --> Average+Noise --> Subsampling
            # X_train, y_train, _, _ = train_data_prep(self.X,self.y,2,2,True)
            # self.X = X_train
            # self.y = y_train
        elif mode=='val':
            self.X = np.load(os.path.join(data_path, "X_train_valid.npy"))
            self.y = np.load(os.path.join(data_path, "y_train_valid.npy"))
            self.person = np.load(os.path.join(data_path, "person_train_valid.npy"))
            self.y-=769
            self.X = self.X[:,:,0:500]  # trim to 500 timesteps

            #Uncomment the below lines for preprocessing
            # _ , _ , X_val,  y_val = train_data_prep(self.X,self.y,2,2,True)
            # self.X = X_val
            # self.y = y_val


        elif mode=='test':
            self.X = np.load(os.path.join(data_path, "X_test.npy"))
            self.y = np.load(os.path.join(data_path, "y_test.npy"))
            self.y-=769
            self.person = np.load(os.path.join(data_path, "person_test.npy"))
            self.X = self.X[:,:,0:500] # trim to 500 timesteps

            #Uncomment the below lines for preprocessing test data
            # X_test  = test_data_prep(self.X)
            # self.X = X_test
        else:
            RuntimeError('Define train or val or test mode!')



        if subject is not None:
            assert type(subject) == int
            self.X = self.X[np.where(self.person==subject)[0],...]
            self.y = self.y[np.where(self.person==subject)[0],...]

        # Random Frequency shift
        self.X, self.y = torch.from_numpy(self.X.astype(np.float32)), torch.from_numpy(self.y.astype(np.int32)).long()
        rfshift = RandomFrequencyShift(sampling_rate=128) #, shift_min=4.0)
        self.X = rfshift(eeg=self.X)['eeg']

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index].unsqueeze(0)   # Comment .unsqueeze() for LSTM() and GRU() !!
        y = self.y[index]

        if self.transform is not None:
            pre_processing = self.transform
            X = pre_processing(X)

        return (X, y)



def train_data_prep(X,y,sub_sample,average,noise):
    np.random.seed(42)
    total_X = None
    total_y = None

    # Generate training and validation indices using random splitting

    ind_valid = np.random.choice(2115, 220, replace=False)
    ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))

    (x_train, x_valid) = X[ind_train], X[ind_valid]
    (y_train, y_valid) = y[ind_train], y[ind_valid]



   ############################TRAIN############################

    # Trimming the data
    X = x_train
    X = X[:,:,0:800]   
    print('Shape of X_train after trimming:',X.shape)

    # Maxpooling the data 
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    total_X = X_max
    total_y = y_train

    # Averaging + noise
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y_train))

    # Subsampling
    for i in range(sub_sample):

        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)

        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y_train))

    x_train, y_train = total_X , total_y

   ############################VAL############################
    # Trimming the data 
    X = x_valid
    X = X[:,:,0:800]

    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    total_X = X_max
    total_y = y_valid

    # Averaging + noise
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y_valid))

    # Subsampling
    for i in range(sub_sample):

        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)

        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y_valid))

    x_valid, y_valid = total_X , total_y

    # Segment width = 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
    return (x_train.squeeze(-1),y_train,x_valid.squeeze(-1),y_valid)

def test_data_prep(X):
    np.random.seed(42)
    total_X = None


    # Trimming the data
    X = X[:,:,0:800]

    # Maxpooling the data
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, 2), axis=3)
    total_X = X_max
    x_test = total_X.reshape(total_X.shape[0], total_X.shape[1], total_X.shape[2], 1)
    return x_test.squeeze(-1)


def smooth_data(data, ws):
    kern = signal.hamming(ws)[None, None, :]
    kern /= kern.sum()
    return signal.convolve(data, kern, mode='same')
