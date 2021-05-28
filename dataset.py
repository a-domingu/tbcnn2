import torch
import os


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels):
            'Initialization'
            self.labels = labels
            #list_IDs is a dictionary whose keys are the 'train' or 'validation' and the values are lists with the files in each category
            self.list_IDs = list_IDs

    def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

    def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]

            # Load data and get label
            X = os.path.join('vector_representation', os.path.basename(ID) + '.txt')
            y = torch.tensor(self.labels[ID])

            return X, y