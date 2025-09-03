from torch.utils.data import DataLoader

def get_dataset_class(name):
    """
    Map a dataset name to its dataset class.
    Supported:
        - 'globsmpl' -> data_loaders.globsmpl_dataset.MotionDataset
    """
    if name == 'globsmpl':
        from data_loaders.globsmpl_dataset import MotionDataset
        return MotionDataset
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, dataset=None):
    """
    Return the collate function for a given dataset.
    - For '*smpl*' datasets, use dataset-specific collate.
    - Otherwise, use the generic tensor collate.
    """
    if 'smpl' in name:
        return dataset.collate_fn
    else:
        raise


def get_dataset(name, split='train', **kwargs):
    """
    Build a dataset instance based on name/split and kwargs.

    Args:
        name (str): Dataset key (e.g., 'globsmpl').
        split (str): Split name ('train', 'val', 'test', etc.).
        **kwargs: Passed through to the underlying loader/dataset.
                  Expected keys for 'globsmpl' include:
                    - data_dir (str): Root directory of data.
                    - normalizer_dir (str), etc., as required downstream.

    Returns:
        torch.utils.data.Dataset
    """
    DATA = get_dataset_class(name)
    if name == 'globsmpl':
        from data_loaders.globsmpl_dataset import AMASSMotionLoader
        fps = 20
        mode = 'train' if 'train' in split else split
        motion_loader = AMASSMotionLoader(
            base_dir = kwargs['data_dir'],
            umin_s = 5.,
            umax_s = 5.,
            ext = '.npz',
            mode = mode,
            fps = fps,
            **kwargs
        )
        dataset = DATA(
            motion_loader = motion_loader,
            split = split,
        )
    else:
        raise

    return dataset


def get_dataset_loader(name, batch_size, split='train', shuffle=True, **kwargs):
    """
    Create a DataLoader for the requested dataset.

    Args:
        name (str): Dataset key (e.g., 'globsmpl').
        batch_size (int): Loader batch size.
        split (str): Split name ('train', 'val', 'test', etc.).
        shuffle (bool): Shuffle batches (usually True for training).
        **kwargs: Forwarded to get_dataset(...).

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = get_dataset(name, split, **kwargs)
    collate = get_collate_fn(name, dataset)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, drop_last=True if 'train' in split else False, collate_fn=collate, pin_memory=True
    )

    return loader