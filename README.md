# Train Learned Planners

This repository contains training code for the paper ["Planning behavior in a recurrent neural network that plays Sokoban"](https://openreview.net/forum?id=T9sB3S2hok), from the ICML 2024 Mechanistic Interpretability Workshop. ([OpenReview](https://openreview.net/forum?id=T9sB3S2hok)) ([arXiv](https://arxiv.org/abs/2407.15421)). It is based on
[CleanRL](https://github.com/vwxyzjn/cleanba).

The [learned-planner repository](https://github.com/AlignmentResearch/learned-planner) lets you download and use the trained neural networks. If you just want to do interpretability, you should go there.

## :rocket: Running Training

First, clone the repo with:

```sh
git clone --recurse-submodules https://github.com/AlignmentResearch/train-learned-planners
# If you have already cloned the repo:
git submodule init
git submodule update --remote
```

We use Docker (on Mac, [Orbstack](https://orbstack.dev)) to easily distribute dependencies. You can get a local
development environment by running `make docker`. If you have a Kubernetes cluster, you can adapt `k8s/devbox.yaml` and
run `make devbox` (or `make cuda-devbox`).

## :gear: Training Commands

The training code expects the [Boxoban levels](https://github.com/google-deepmind/boxoban-levels) in
`/opt/sokoban_cache/boxoban-levels-master`, but it is possible to change that path. You can download them using:

```sh
BOXOBAN_CACHE="/opt/sokoban_cache/"  # change if desired
mkdir -p "$BOXOBAN_CACHE"
git clone https://github.com/google-deepmind/boxoban-levels \
  "$BOXOBAN_CACHE/boxoban-levels-master"
```

The launcher scripts for the final runs are numbered [`061_pfinal2`](./experiments/sokoban/061_pfinal2.py) and above.

### Training the ConvLSTM (DRC)

For DRC(3, 3):

```sh
python -m cleanba.cleanba_impala --from-py-fn=cleanba.config:sokoban_drc33_59 \
  "train_env.cache_path=$BOXOBAN_CACHE" \
   "eval_envs.valid_medium.cache_path=$BOXOBAN_CACHE"
```

For DRC(D, N) (e.g. DRC(1, 1)):

```sh
D=1
N=1
python -m cleanba.cleanba_impala --from-py-fn=cleanba.config:sokoban_drc33_59 \
  "train_env.cache_path=$BOXOBAN_CACHE" \
  "eval_envs.valid_medium.cache_path=$BOXOBAN_CACHE" \
  net.n_recurrent=$D net.repeats_per_step=$N
```

### Training the ResNet

```sh
python -m cleanba.cleanba_impala --from-py-fn=cleanba.config:sokoban_resnet_59 \
  "train_env.cache_path=$BOXOBAN_CACHE" \
  "eval_envs.valid_medium.cache_path=$BOXOBAN_CACHE"
```

## :package: Local Install (May Fail)

From inside your Python 3.10 local environment, run:

```sh
make local-install
```

If you're not on Linux, Python 3.10, and x86_64, you will get the following error:

```
ERROR: envpool-0.8.4-cp310-cp310-linux_x86_64.whl is not a supported wheel on this platform.
```

You can still use non-envpool environments by using `BoxobanConfig` and `SokobanConfig` (in
[`cleanba/environments.py`](cleanba/environments.py)).

## :hammer_and_wrench: Development

- **Experiment lists:** All the experiments we ran to debug and tune hyperparameters are under `experiments/`. Each
  experiment launches jobs in a Kubernetes cluster.
- **Tests:** Run `make mactest` to run all the tests expected to succeed on a local machine.
- **Linting:** Run `make lint format typecheck` to lint, format, and typecheck the code.

## :bookmark_tabs: Citation

If you use this code, please cite our work:

```bibtex
@inproceedings{garriga-alonso2024planning,
    title={Planning behavior in a recurrent neural network that plays Sokoban},
    author={Adri{\`a} Garriga-Alonso and Mohammad Taufeeque and Adam Gleave},
    booktitle={ICML 2024 Workshop on Mechanistic Interpretability},
    year={2024},
    url={https://openreview.net/forum?id=T9sB3S2hok}
}
```
