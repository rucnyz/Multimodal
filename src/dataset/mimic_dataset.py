from torch.utils.data import Dataset


class Mimic_Dataset(Dataset):
    # 把dataroot改成mimic的目录
    def __init__(self, name, args, token_to_ix = None, dataroot = 'D:\\BaiduNetdiskDownload\\mimic\\mimic3'):
        super(Mimic_Dataset, self).__init__()
        self.token_to_ix = token_to_ix
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.args = args
        self.dataroot = dataroot