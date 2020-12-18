# Yotta - Project 2

Project Nº2: Deep Learning: French Amendments Topic Analysis

## Getting started

### Requirements

Following tools must be install to setup this project:

- `python >= 3.8`
- `poetry >= 1.1`
- `git lfs`
- `cuda == 10.2 or cpu`

_You can't push to HuggingFace git, if you want to, please contact us_

### Setup environment

Following command lines could be used to setup the project.

```
By SSH
$ git clone git@gitlab.com:yotta-academy/mle-bootcamp/projects/dl-projects/fall-2020/project_2_vdv_tb_eds.git
or By https
$ git clone https://gitlab.com/yotta-academy/mle-bootcamp/projects/dl-projects/fall-2020/project_2_vdv_tb_eds.git
$ cd amendements_analysis/
$ poetry install  # Install virtual environment with packages from pyproject.toml file
```

### Run script

_If you don't have already download the amendments, it will downlaod it from the french website_

```
$ poetry shell
```

Cmd to train the language model Camembert-base

```
$ train_lm --publish  <True or False>
```

Cmd to train the cluster model to get the topics

```
$ cluster_train
```

Cmd to predict a topic from a cluster (specify it in code for now)

```
$ cluster_predict
```
