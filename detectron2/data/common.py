# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import pickle
import random
import torch.utils.data as data

from os.path import basename
from natsort import natsorted

from detectron2.utils.serialize import PicklableWrapper

__all__ = ["MapDataset", "DatasetFromList", "AspectRatioGroupedDataset", "stack_dicts"]


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func, is_stack: bool):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work
        self._is_stack = is_stack
        if self._is_stack:
            self._stack_size = len(self._dataset[0])

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            if self._is_stack:
                z_data = [None] * self._stack_size
                map_succeed = True
                for z in range(self._stack_size):
                    data = self._map_func(self._dataset[cur_idx][z])
                    if data is not None:
                        z_data[z] = data
                    else:
                        map_succeed = False

                if map_succeed:
                    self._fallback_candidates.add(cur_idx)
                    return z_data

            else:
                data = self._map_func(self._dataset[cur_idx])
                if data is not None:
                    self._fallback_candidates.add(cur_idx)
                    return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


def gather_stack_dicts(lst: list, stack_size: int = 11, ext: str = ".png", sep: str = "F"):

    stacks_dict = {basename(lst[0]['file_name']).split(sep, 1)[0] : [lst[0]]}
    nb_img = len(lst)
    for i in range(1, nb_img):
        i_stack = basename(lst[i]['file_name']).split(sep, 1)[0]
        if i_stack in stacks_dict:
            stacks_dict[i_stack].append(lst[i])
        else:
            stacks_dict[i_stack] = [lst[i]]

        if len(stacks_dict[i_stack]) == stack_size:
            stacks_dict[i_stack] = natsorted(stacks_dict[i_stack], key = lambda d : basename(d['file_name'])[:-len(ext)])

    # Create the list of the stacked dictionaries & verify if it is well constructed
    logger = logging.getLogger(__name__)

    z_lst = list(stacks_dict.values())
    nb_stacks = len(z_lst)
    logger.info("Number of stacks: {}".format(nb_stacks))
    cnt_img = 0
    cnt_too_big = 0
    cnt_too_small = 0

    for s in range(nb_stacks):
        cnt_img += len(z_lst[s])
        if len(z_lst[s]) == stack_size:
            stack_sorted = True
            for z in range(stack_size):
                if basename(z_lst[s][z]['file_name']).split(sep, 1)[1][:-len(ext)] != str(z):
                    stack_sorted = False
            if not stack_sorted:
                logger.warning("Stack {} is not sorted ({})".format(s, basename(z_lst[s][0]['file_name']).split(sep, 1)[0]))
        elif len(z_lst[s]) > stack_size:
            cnt_too_big += 1
        elif len(z_lst[s]) < stack_size:
            cnt_too_small +=1
            
    if cnt_img != nb_img:
        logger.warning("There are {} images in total, which is not expected ({})".format(cnt_img, nb_img))
    else:
        logger.info("There are {} images in total".format(cnt_img))

    assert cnt_too_big == 0, "{} stacks have a bigger size than expected ({})".format(cnt_too_big, stack_size)
    assert cnt_too_small == 0, "{} stacks have a smaller size than expected ({})".format(cnt_too_small, stack_size)
    logger.info("All stacks have {} images".format(stack_size))

    return z_lst
    

class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(
            self, 
            lst: list,
            cfg,
            copy: bool = True,
            serialize: bool = True
        ):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        if cfg.DATALOADER.IS_STACK:
            self._lst = gather_stack_dicts(lst, cfg.INPUT.STACK_SIZE, cfg.INPUT.EXTENSION, cfg.INPUT.SLICE_SEPARATOR)
        else:
            self._lst = lst
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger = logging.getLogger(__name__)
            logger.info(
                "Serializing {} elements to byte tensors and concatenating them all ...".format(
                    len(self._lst)
                )
            )
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def __getitem__(self, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            return pickle.loads(bytes)
        elif self._copy:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]


class AspectRatioGroupedDataset(data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]
