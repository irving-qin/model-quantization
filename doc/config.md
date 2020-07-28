
## Classification

Default naming pattern is `config` + `method` + `phase` + `dataset` + `precision` + `network`.

where `phase` contains `eval` for evaluation; `train-stratch` for training without pretrained model; `finetuning` means training with pretrained model as the initilization.

`dataset` can be chosen from `imagenet` for the imagenet dataset; `dali` for the imagenet dataset wrappered in nvidia dali for fast preprocessing; `cifar10` for CIFAR10 dataset; `cifar100` for CIFAR100 dataset and `fake` for fake images for testing.

For example, `config.dorefa.eval.imagenet.fp.resnet18` inidcates to evaluate the ResNet-18 network on imagenet dataset with full precision.

Run `bash train.sh config.dorefa.eval.imagenet.fp.resnet18` to test the PreBN ResNet-18 on imagenet. Expected accuracy: Top-1(70.1) and Top-5(89.3).

## Super Resolution

### For developers

1. Do not include any user/name/folder/path with respect to one's own machine in any config file and scripts. Inlcude those in `.env` file with enviroment vairables. Relative path is suggested.

2. In the configuration `config.xx` files, all options are advised to have one only space ahead, replace with `#` to comment it if required. Do not add extra space to keep left alignment.
