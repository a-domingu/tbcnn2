import os
import pandas as pd
import torch


class GetTargets():

    def __init__(self, path):
        self.targets_path = path
        self.files = self.files_iterator()

    def files_iterator(self):
        # iterates through the directory, identifies the .csv files and yields them
        for (dirpath, dirnames, filenames) in os.walk(self.targets_path):
            for filename in filenames:
                if filename.endswith('.csv'):
                    filename_full_path = os.path.join(dirpath, filename)
                    yield filename_full_path


    def files_opener(self):
        #opens the files and parses the data into a pandas dataframe
        for filename in self.files:
            file_data = pd.read_csv(filename, delimiter = ',')
            yield (file_data, filename)

    def df_iterator(self):
        # iterates through each dataframe and it calls the 'get_target' method to get the target for
        # each dataframe
        file_data = self.files_opener()
        targets = {}
        for df, filename in file_data:
            target = self.get_target(df)
            targets[filename] = target
        return targets

    def get_target(self, df):
        generator_column = df['Generator']
        if generator_column.sum() >= 1:
            target = torch.ones(1)
            return target
        else:
            target = torch.zeros(1)
            return target








