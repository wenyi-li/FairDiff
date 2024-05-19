import os
import random

from torch.utils.data import DataLoader, random_split
def get_train_val_test_datasets(dataset, train_ratio, val_ratio):
    assert (train_ratio + val_ratio) <= 1
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    return train_set, val_set, test_set


def get_train_val_test_loaders(dataset, train_ratio, val_ratio, train_batch_size, val_test_batch_size, num_workers):
    train_set, val_set, test_set = get_train_val_test_datasets(dataset, train_ratio, val_ratio)

    train_loader = DataLoader(train_set, train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, val_test_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, val_test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def split(path="/DATA_EDS2/liwy/datasets/MedicalImage/point_cloud_outline-100/White", train_ratio=0.9, test_ratio=0.1):
    list_all = os.listdir(path)

    random.shuffle(list_all)
    list_all_sort = sorted(list_all, key=lambda x: (int(x.split('.')[0])))
    # i = 1
    # train_list = []
    # test_list = []
    # for file in list_all:
    #     if i % 2 == 1:
    #         train_list.append(file)

    length = len(list_all)
    train_list = list_all[:int(train_ratio*length)]
    train_list = sorted(train_list, key=lambda x: (int(x.split('.')[0])))
    test_list = list_all[int(train_ratio*length):]
    test_list = sorted(test_list, key=lambda x: (int(x.split('.')[0])))
    # with open("./train.txt", "w") as f:
    #     for item in train_list:
    #         f.write(item + "\n")
    # with open("./test.txt", "w") as f:
    #     for item in test_list:
    #         f.write(item + "\n")
    with open("/data22/DISCOVER_summer2023/zhanggy2308/Medical-Image/DDPM-outline100/data/White.txt", "w") as f:
        for item in list_all_sort:
            f.write(item + "\n")
if __name__ == '__main__':
    split()