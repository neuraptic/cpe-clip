# About data splits

You should place in this directory the splits folders downloaded from the shared Google Drive link, as stated in the `README.md` file in the root of this repository.

These splits are based on the original splits of the FSCIL benchmark, following the same structure as the rest of the state-of-the-art methods.

The splits can be found in the following [Google Drive shared folder](https://drive.google.com/drive/folders/1nVZJFySisbrq0t8ReGX6Yyg2MOtAkCdQ?usp=drive_link). You must download just the folders relative to this paper, so:
- [`cifar100/cpe-clip-cifar100-splits.zip`](https://drive.google.com/file/d/1TpDUpUoy6pHUShbmnaRYFs1TfydV65-e/view?usp=drive_link).
- [`cub200/cpe-clip-cub200-splits.zip`](https://drive.google.com/file/d/1PumwrWQCNZTBbgW6bbZ6NDD4xyA0wGfH/view?usp=drive_link).
- [`miniimagenet/cpe-clip-miniimagenet-splits.zip`](https://drive.google.com/file/d/1eoxn4gAJ_3823Xh4yDzhVoklANFOYgkR/view?usp=drive_link).

Once you have downloaded the `.zip` files, place them in the `/splits` folder and extract them.

```bash
unzip cpe-clip-cifar100-splits.zip
unzip cpe-clip-cub200-splits.zip
unzip cpe-clip-miniimagenet-splits.zip
```

The structure of the directory should be as follows:

```
splits
├── cifar100
│   ├── base_class.pkl
│   ├── class_to_id.pkl
│   ├── exp0.pkl
│   ├── ...
│   ├── exp7.pkl
│   ├── session_1.txt
│   ├── ...
│   ├── session_9.txt
│   ├── test_1.txt
│   ├── ...
│   └── test_9.txt
├── cub200
│   ├── base_class.pkl
│   ├── class_to_id.pkl
│   ├── exp0.pkl
│   ├── ...
│   ├── exp9.pkl
│   ├── session_1.txt
│   ├── ...
│   ├── session_11.txt
│   ├── test_1.txt
│   ├── ...
│   └── test_11.txt
└── miniimagenet
    ├── base_class.pkl
    ├── class_to_id.pkl
    ├── exp0.pkl
    ├── ...
    ├── exp7.pkl
    ├── session_1.txt
    ├── ...
    ├── session_9.txt
    ├── test_1.txt
    ├── ...
    └── test_9.txt
```

The `.pkl` files contain the already splited data with the images and labels, and the `.txt` files contain the indexes of the images to be used in the training and testing phases.

Note that, as in the original FSCIL benchmark, the datasets are already split into different sessions and each session has a pre-defined number of classes and examples. These splits are as follows:

- CIFAR100 (5-way 5-shot incremental task)
    - Base session: 60 classes.
    - 8 incremental sessions: 5 classes with 5 examples each.
- CUB200 (10-way 5-shot incremental task)
    - Base session: 100 classes.
    - 10 incremental sessions: 10 classes with 5 examples each.
- miniImageNet (5-way 5-shot incremental task)
    - Base session: 100 classes.
    - 8 incremental sessions: 5 classes with 5 examples each.

[!IMPORTANT]
Remember that you also must follow the instructions provided in the [data README](/data/README.md) about how to download the datasets. Without both steps, the model will not be able to train.