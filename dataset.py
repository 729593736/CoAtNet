import tarfile
import pickle
from typing import List, Optional, Union
from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 字典关键字
key_num_cases_per_batch = b'num_cases_per_batch'
key_label_names = b'label_names'
key_num_vis = b'num_vis'
key_batch_label = b'batch_label'
key_labels = b'labels'
key_data = b'data'
key_filenames = b'filenames'

# 文件路径
# path_tar = "./dataset/cifar-10-python.tar.gz"
# path_dataset = './dataset'
path_basedir = "./cifar-10-batches-py"
path_data_meta = path_basedir + '/batches.meta'
path_data_batch = path_basedir + '/data_batch_'
path_test_batch = path_basedir + '/test_batch'

def untar(src: str, dst_dir: str) -> None:
    """
    解压*.tar.gz文件
    ------
    - param src: 压缩文件路径
    - param dst_dir: 压缩文件提取存储目录路径
    - return: None
    """
    with tarfile.open(src) as fp:
        names = fp.getnames()
        for name in names:
            fp.extract(name, dst_dir)


def unpickle(filepath: str) -> dict:
    """
    反序列化python字典数据
    --------
    - param filepath: 需要读取的序列化文件路径
    - return: 字典对象 {标签: numpy矩阵格式的图像}
    """
    with open(filepath, 'rb') as fo:
        label2arr = pickle.load(fo, encoding='bytes')
    return label2arr


class DatasetBuilder:
    def __init__(self, filepath: str, image_shape: tuple) -> None:
        """
        - param filepath: 需要读取的序列化文件路径
        - param image_shape: 图像数据的形状(channels, height, width)
        """
        self.filepath = filepath
        self.image_shape = image_shape

        self.data_dict = unpickle(filepath)
        self.label_batch = self.data_dict[key_batch_label]
        self.labels = self.data_dict[key_labels]
        self.data = self.data_dict[key_data]
        self.filenames = self.data_dict[key_filenames]
        self.num_samples = len(self.labels)

    def build(self, data_meta: dict) -> pd.DataFrame:
        """
        使用DataFrame格式构建数据集
        -----
        - param data_meta: 数据集信息
        - return: 返回DataFrame格式的数据集
        """
        labels_name_set = data_meta[key_label_names]
        # 字节转字符串
        labels_name_set = [
            str(name, encoding='utf-8') for name in labels_name_set
        ]
        # # (num, c, h, w) => (num, h, w, c)
        # data_list = list(
        #     self.data.reshape(
        #         (self.num_samples,
        #          *self.image_shape)).transpose(0, 2, 3, 1).astype(np.float))
        # (n, c, h, w)
        data_list = list(
            self.data.reshape(
                (self.num_samples,
                 *self.image_shape)).astype(np.float32))
        data_list = [np.array(data) / 255.0 for data in data_list]

        dataset = pd.DataFrame()
        dataset["label_batch"] = [str(self.label_batch, encoding='utf-8')[-6]
                                  ] * self.num_samples
        dataset["labels"] = [int(label) for label in self.labels]
        dataset['labels_name'] = dataset['labels'].apply(
            lambda label: labels_name_set[label])
        dataset['data'] = data_list
        dataset['filenames'] = [
            str(fn, encoding='utf-8') for fn in self.filenames
        ]
        return dataset


class Dataset:
    """
    数据读取的方式
    """
    def __init__(self, dataset: pd.DataFrame, transform=None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        label_batch, label, label_name, data, filename = self.dataset.iloc[
            index].tolist()
        if self.transform is not None:
            data = self.transform(data)

        return filename, data, label


def getDataLoader(train_dataIndex: Union[int, List[int]] = None,
                  is_train_dataset: bool = True,
                  batch_size: int = 1,
                  shuffle: bool = True) -> DataLoader:


    # 图像维度
    image_shape = (3, 32, 32)
    data_meta = unpickle(path_data_meta)
    if is_train_dataset:
        if isinstance(train_dataIndex, int):
            dataset_batches = DatasetBuilder(
                path_data_batch + str(train_dataIndex),
                image_shape).build(data_meta)
        elif isinstance(train_dataIndex, list):
            dataset_batches = [
                DatasetBuilder(path_data_batch + str(idx),
                               image_shape).build(data_meta)
                for idx in train_dataIndex
            ]
            dataset_batches = pd.concat(dataset_batches,
                                        axis=0,
                                        ignore_index=True)
        else:
            raise Exception('类型不匹配')
            pass
    else:
        dataset_batches = DatasetBuilder(path_test_batch,
                                         image_shape).build(data_meta)
        pass
    dataset = Dataset(dataset_batches)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

def getClassesName():
    data_meta = unpickle(path_data_meta)
    classes_name = [str(name) for name in data_meta[b'label_names']]
    return classes_name
