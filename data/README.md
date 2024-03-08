# About data

You should place in this directory the data folders downloaded from the shared Google Drive link, as stated in the `README.md` file in the root of this repository.

Those folders contain the datasets used in the paper with the exception of CIFAR100, that only needs the dataset splits `.zip` file (see [splits README](/splits/README.md) for more information).

The datasets can be found in the following [Google Drive shared folder](https://drive.google.com/drive/folders/1nVZJFySisbrq0t8ReGX6Yyg2MOtAkCdQ?usp=drive_link). You must download just the datasets relative to this paper, so:
- [`cub200/cub200.zip`](https://drive.google.com/file/d/1PumwrWQCNZTBbgW6bbZ6NDD4xyA0wGfH/view?usp=drive_link)
- [`miniimagenet/miniimagenet.zip`](https://drive.google.com/file/d/1eoxn4gAJ_3823Xh4yDzhVoklANFOYgkR/view?usp=drive_link)

With the downloaded `.zip` files already placed in the `/data` folder, you need to extract them. You can do it as follows:

```bash
unzip cub200.zip
unzip miniimagenet.zip
```

Once you have extracted both datasets, they will be ready to be used for training.

[!IMPORTANT]
Remember that you also must follow the instructions provided in the [splits README](/splits/README.md) to download the datasets splits needed for the FSCIL setting. Without both steps, the model will not be able to train.