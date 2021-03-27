from torch.utils.data import Dataset
import numpy as np

class LoadData(Dataset):
    def __init__(self, indexes, data, label):
        super(LoadData, self).__init__()
        self.DATA = data
        self.LABEL = label
        self.namelist = indexes


    def __getitem__(self, index):
        name = self.namelist[index]
        data = self.DATA.loc[name].values.astype(np.float32)
        label = self.LABEL.loc[name].values[0].astype(np.int)
        return data, label



    def __len__(self):
        return len(self.namelist)
