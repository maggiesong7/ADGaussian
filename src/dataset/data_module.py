import numpy as np
import torch

from pytorch_lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Callable

from . import get_dataset
from src.dataset.types import Stage


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)
    
    
DatasetShim = Callable[[Dataset, Stage], Dataset]


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_cfg,
        data_loader_cfg,
        global_rank,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
    ):
        super().__init__()
        
        self.data_cfg = data_cfg
        self.data_loader_cfg = data_loader_cfg
        
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank

    def get_generator(self, loader_cfg):
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator
    
    def train_dataloader(self):
        train_ds = get_dataset("train", self.data_cfg)
        train_ds = self.dataset_shim(train_ds, "train")
        return DataLoader(
            train_ds,
            batch_size=self.data_loader_cfg.train.batch_size,
            drop_last=True,
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.get_generator(self.data_loader_cfg.train),
            shuffle=not isinstance(train_ds, IterableDataset),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        val_ds = get_dataset("val", self.data_cfg)
        val_ds = self.dataset_shim(val_ds, "val")
        return DataLoader(
            val_ds,
            batch_size=self.data_loader_cfg.val.batch_size,
            drop_last=False,
            num_workers=self.data_loader_cfg.val.num_workers,
            generator=self.get_generator(self.data_loader_cfg.val),
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self):
        test_ds = get_dataset("test", self.data_cfg)
        test_ds = self.dataset_shim(test_ds, "test")
        return DataLoader(
            test_ds,
            batch_size=self.data_loader_cfg.test.batch_size,
            drop_last=False,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.get_generator(self.data_loader_cfg.test),
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

