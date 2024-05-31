# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import copy

import numpy as np
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset

class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        
    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        
        self.samples = []
        
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }
        
        for sample in tqdm(self.dataset, desc="Preprocessing dataset"):
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

                
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)


class ConcatDatasetNumpy(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        # self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []
        self.input_ids = np.concatenate([x for x in dataset])
        print(f'data set input id size = {self.input_ids.shape[0]}')
        # labels = np.concatenate([x['labels'] for x in dataset])
        # attention_mask = np.concatenate([x['attention_mask'] for x in dataset])

        # for i in tqdm(range(input_ids.shape[0] // chunk_size)):
        #     self.samples.append({
        #         'input_ids': input_ids[i * chunk_size: (i+1) * chunk_size],
        #         'labels': labels[i * chunk_size: (i+1) * chunk_size],
        #         'attention_mask': attention_mask[i * chunk_size: (i+1) * chunk_size],
        #     })
    def __getitem__(self, idx):
        bos = idx * self.chunk_size
        input_ids = self.input_ids[bos: bos + self.chunk_size]
        labels = copy.deepcopy(input_ids)
        data = {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': np.ones_like(input_ids, dtype=np.float32),
            }
        return data

    def __len__(self):
        return self.input_ids.shape[0] // self.chunk_size