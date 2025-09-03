import torch

from typing import List, Dict, Optional
from torch import Tensor
from torch.utils.data import default_collate


def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_motion(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    other_keys = [key for key in keys if key not in ["x"]]
    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}

    x = collate_tensor_with_padding([x["x"] for x in lst_elements])
    if device is not None:
        x = x.to(device)

    batch["x"] = x.transpose(1, 2)

    if "length" in batch:
        batch["mask"] = length_to_mask(batch["length"], device=x.device)

    return batch
