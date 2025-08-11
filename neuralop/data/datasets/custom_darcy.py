import logging
import os
from pathlib import Path
from typing import Union, List
from copy import deepcopy

from torch.utils.data import DataLoader
from torch.nn.functional import pad

from .tensor_dataset import TensorDataset
from .pt_dataset import PTDataset
import gdown

logger = logging.Logger(logging.root.level)

class CustomDarcyDataset(PTDataset):
    """
    DarcyDataset stores data generated according to Darcy's Law.
    Input is a coefficient function and outputs describe flow. 

    Attributes
    ----------
    train_db: torch.utils.data.Dataset of training examples
    test_db:  ""                       of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """
    def __init__(self,
                 root_dir: Union[Path, str],
                 dataset_name: str,
                 n_train: int,
                 n_tests: List[int],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: List[int]=[64, 1024],
                 encode_input: bool=False, 
                 encode_output: bool=True, 
                 encoding="channel-wise",
                 channel_dim=1,
                 subsampling_rate=None):

        """DarcyDataset

        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
        n_train : int
            number of train instances
        n_tests : List[int]
            number of test instances per test dataset
        batch_size : int
            batch size of training set
        test_batch_sizes : List[int]
            batch size of test sets
        train_resolution : int
            resolution of data for training set
        test_resolutions : List[int], optional
            resolution of data for testing sets, by default [16,32]
        encode_input : bool, optional
            whether to normalize inputs in provided DataProcessor,
            by default False
        encode_output : bool, optional
            whether to normalize outputs in provided DataProcessor,
            by default True
        encoding : str, optional
            parameter for input/output normalization. Whether
            to normalize by channel ("channel-wise") or 
            by pixel ("pixel-wise"), default "channel-wise"
        input_subsampling_rate : int or List[int], optional
            rate at which to subsample each input dimension, by default None
        output_subsampling_rate : int or List[int], optional
            rate at which to subsample each output dimension, by default None
        channel_dim : int, optional
            dimension of saved tensors to index data channels, by default 1
        """

        # convert root dir to Path
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        # List of resolutions needed for dataset object
        resolutions = set(test_resolutions + [train_resolution])

        # We store data at these resolutions on the Zenodo archive
        available_resolutions = [64, 256, 1024]
        for res in resolutions:
            assert res in available_resolutions, f"Error: resolution {res} not available"

        # download darcy data from zenodo archive if passed
        files_to_download = []
        already_downloaded_files = [x.name for x in root_dir.iterdir()]
        for res in resolutions:
            if f"{dataset_name}_train_{res}.pt" not in already_downloaded_files:  
                files_to_download.append(f"{dataset_name}_train_{res}.pt")
            if f"{dataset_name}_test_{res}.pt" not in already_downloaded_files:
                files_to_download.append(f"{dataset_name}_test_{res}.pt")
        file_ids = {
            'darcy_ZD_PWC_train_64.pt': '1hQzdQpudGTBZ7x0nURPzlV5DTcqcb1sm',
            'darcy_ZD_PWC_test_64.pt': '1qsstWhpdYRkj7dTSeVpfDFWknj0LpCPw',
            'darcy_ZD_PWC_train_256.pt': '1RYATFqGmVOKAUNAXVKf2P5Cdw7GYsoSe',
            'darcy_ZD_PWC_test_256.pt': '1YcJDtuJsNG0pLPBYKDLgtsxSlcEsUrde',
            'darcy_ZD_PWC_train_1024.pt': '1OLtn2-u18P_wz6EFQfLKpcvAvyJn496U',
            'darcy_ZD_PWC_test_1024.pt': '1W2ToRtFSms-ncoOeW545UiM1ZfMeSmiz',
        }
        for file in files_to_download:
            # gdown.download(id=file_ids[file], output=root_dir)
            gdown.download(id=file_ids[file], output=str(root_dir / file))
            
        # once downloaded/if files already exist, init PTDataset
        super().__init__(root_dir=root_dir,
                         dataset_name=dataset_name,
                         n_train=n_train,
                         n_tests=n_tests,
                         batch_size=batch_size,
                         test_batch_sizes=test_batch_sizes,
                         train_resolution=train_resolution,
                         test_resolutions=test_resolutions,
                         encode_input=encode_input,
                         encode_output=encode_output,
                         encoding=encoding,
                         channel_dim=channel_dim,
                         input_subsampling_rate=subsampling_rate,
                         output_subsampling_rate=subsampling_rate)

def load_darcy_flow(root_dir,
                    dataset_name,
                    n_train,
                    n_tests,
                    batch_size,
                    test_batch_sizes,
                    train_resolution,
                    test_resolutions,
                    encode_input=False,
                    encode_output=True,
                    encoding="channel-wise",
                    channel_dim=1,
                    **kwargs):

    dataset = CustomDarcyDataset(root_dir=root_dir,
                                 dataset_name=dataset_name,
                                 n_train=n_train,
                                 n_tests=n_tests,
                                 batch_size=batch_size,
                                 test_batch_sizes=test_batch_sizes,
                                 train_resolution=train_resolution,
                                 test_resolutions=test_resolutions,
                                 encode_input=encode_input,
                                 encode_output=encode_output,
                                 channel_dim=channel_dim,
                                 encoding=encoding)
    
    if kwargs.get('decompose_dataset', False):
        subdomain_datasets = decompose_darcy_dataset(dataset, **kwargs)

        train_loader_list = []
        test_loaders_list = []
        data_processor_list = []
        for subdataset in subdomain_datasets:
            train_loader_list.append(DataLoader(subdataset.train_db,
                                                batch_size=batch_size,
                                                num_workers=1,
                                                pin_memory=True,
                                                persistent_workers=False,))
            test_loaders = {}
            for res, test_bsize in zip(test_resolutions, test_batch_sizes):
                test_loaders[res] = DataLoader(subdataset.test_dbs[res],
                                               batch_size=test_bsize,
                                               shuffle=False,
                                               num_workers=1,
                                               pin_memory=True,
                                               persistent_workers=False,)
            test_loaders_list.append(test_loaders)
            data_processor_list.append(subdataset.data_processor)
        return train_loader_list, test_loaders_list, data_processor_list
    else:
        train_loader = DataLoader(dataset.train_db,
                                  batch_size=batch_size,
                                  num_workers=1,
                                  pin_memory=True,
                                  persistent_workers=False,)
        
        test_loaders = {}
        for res,test_bsize in zip(test_resolutions, test_batch_sizes):
            test_loaders[res] = DataLoader(dataset.test_dbs[res],
                                           batch_size=test_bsize,
                                           shuffle=False,
                                           num_workers=1,
                                           pin_memory=True,
                                           persistent_workers=False,)
        
        return train_loader, test_loaders, dataset.data_processor

# --------------------------------------------------

def decompose_darcy_dataset(
    dataset,
    subdomain_size: int = 32,
    padding: int = 8,
    stride: int = None,
    **kwargs
) -> List:
    """
    Splits each sample in the DarcyDataset into smaller patches.
    Supports overlapping subdomains using custom stride.
    
    Args:
        dataset: DarcyDataset
        subdomain_size: size of subdomain (e.g., 32)
        padding: number of padding cells for coefficient input
        stride: stride between patches (default: subdomain_size for non-overlapping)

    Returns:
        A list of DarcyDataset objects, one per subdomain position.
    """
    if stride is None or stride == 0:
        stride = subdomain_size
        enforce_tiling = True
    else:
        enforce_tiling = False

    def decompose_tensor_pair(x_all, y_all):
        N, C, H, W = x_all.shape
        
        if enforce_tiling:
            assert H % subdomain_size == 0, "Grid size must be divisible by subdomain size when stride=0"
        else:
            assert (H - subdomain_size) % stride == 0, "Stride must evenly divide domain size - subdomain_size"

        sub_xs = []
        sub_ys = []

        for top in range(0, H - subdomain_size + 1, stride):
            for left in range(0, W - subdomain_size + 1, stride):
                bottom = top + subdomain_size
                right = left + subdomain_size

                pad_top = max(padding - top, 0)
                pad_left = max(padding - left, 0)
                pad_bottom = max(bottom + padding - H, 0)
                pad_right = max(right + padding - W, 0)

                x_crop = x_all[:, :, max(top - padding, 0):min(bottom + padding, H),
                                     max(left - padding, 0):min(right + padding, W)]
                x_crop = pad(x_crop, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)

                y_crop = y_all[:, :, top:bottom, left:right]

                sub_xs.append(x_crop)
                sub_ys.append(y_crop)

        return sub_xs, sub_ys

    # Decompose full dataset
    train_x_subs, train_y_subs = decompose_tensor_pair(dataset._train_db.x, dataset._train_db.y)
    test_x_subs = {}; test_y_subs = {}
    for resol, test_db in dataset.test_dbs.items():
        test_x_subs[resol], test_y_subs[resol] = decompose_tensor_pair(test_db.x, test_db.y)

    # # Check matching lengths
    # assert len(train_x_subs) == len(train_y_subs) == len(test_x_subs) == len(test_y_subs), \
    #     "Mismatch in number of subdomains"

    # Create one dataset per subdomain location
    subdomain_datasets = []
    for i in range(len(train_x_subs)):
        sub_dataset = deepcopy(dataset)
        sub_dataset._train_db = TensorDataset(train_x_subs[i], train_y_subs[i])
        test_dbs = {}
        for resol in test_x_subs.keys():
            test_dbs[resol] = TensorDataset(test_x_subs[resol][i], test_y_subs[resol][i])
        sub_dataset._test_dbs = test_dbs
        subdomain_datasets.append(sub_dataset)

    return subdomain_datasets