from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder
from dataset import DatasetFromFolder2
from torch.utils.data import DataLoader


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform():
    return Compose([
        ToTensor(),
    ])



def get_patch_training_set(upscale_factor, patch_size):
    root_dir = "/userhome/mq/data/CAVEdata12/"
    train_dir1 = join(root_dir, "train/X")
    train_dir2 = join(root_dir, "train/Y")
    test_dir3 = join(root_dir, "train/X_blur")

    return DatasetFromFolder(train_dir1,train_dir2,test_dir3,upscale_factor, patch_size, input_transform=input_transform())


def get_test_set():
    root_dir = "/userhome/mq/data/CAVEdata12/"
    test_dir1 = join(root_dir, "test/X")
    test_dir2 = join(root_dir, "test/Y")
    test_dir3 = join(root_dir, "test/Z")

    return DatasetFromFolder2(test_dir1,test_dir2,test_dir3, input_transform=input_transform())


# if __name__ == '__main__':
#     train_set = get_patch_training_set(2)
#     test_set = get_test_set(2)
#     training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=8,
#                                       shuffle=False)
#     testing_data_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=8,
#                                      shuffle=False)
#     for iteration, batch in enumerate(training_data_loader, 1):
#         Y, X_1, X_2, X = batch[0], batch[1], batch[2], batch[3]
#         print("X", X.shape)
#         print("X_1", X_1.shape)
#         print("X_2", X_2.shape)
#         print("Y", Y.shape)
#
#     for batch in testing_data_loader:
#         Y, X_1, X_2, X = batch[0], batch[1], batch[2], batch[3]
#         print("X", X.shape)
#         print("X_1", X_1.shape)
#         print("X_2", X_2.shape)
#         print("Y", Y.shape)

