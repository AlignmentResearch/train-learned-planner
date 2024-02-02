ARG CUDA_VERSION=12.2.2-cudnn8
ARG CUDA_BASE=ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${CUDA_BASE} as envpool-environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y golang-1.18 git python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/lib/go-1.18/bin:/root/go/bin:$PATH
RUN go install github.com/bazelbuild/bazelisk@v1.19.0 && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@v0.0.0-20231115204819-d4c9dccdfbb1

ENV USE_BAZEL_VERSION=6.4.0
RUN bazel version
WORKDIR /app

FROM envpool-environment as envpool

COPY third_party/envpool .
# Abort if repo is dirty
RUN echo "$(git status --porcelain --ignored=traditional)" \
    && if ! { [ -z "$(git status --porcelain --ignored=traditional)" ] \
    ; }; then exit 1; fi

RUN make bazel-release

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${CUDA_BASE} as dependencies

ARG APPLICATION_NAME
ARG USERID=1001
ARG GROUPID=1001
ARG USERNAME=dev

ENV GIT_URL="https://github.com/AlignmentResearch/${APPLICATION_NAME}"

ENV DEBIAN_FRONTEND=noninteractive
MAINTAINER Adrià Garriga-Alonso <adria@far.ai>
LABEL org.opencontainers.image.source=${GIT_URL}

RUN apt-get update -q \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    # essential for running. GCC is for Torch triton
    git git-lfs tini build-essential \
    # essential for testing
    libgl-dev libglib2.0-0 zip make \
    # devbox niceties
    curl vim tmux less sudo \
    # CircleCI
    ssh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tini: reaps zombie processes and forwards signals
ENTRYPOINT ["/usr/bin/tini", "--"]

# Simulate virtualenv activation
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python3 -m venv "${VIRTUAL_ENV}" --system-site-packages \
    && addgroup --gid ${GROUPID} ${USERNAME} \
    && adduser --uid ${USERID} --gid ${GROUPID} --disabled-password --gecos '' ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p "/workspace" \
    && chown -R ${USERNAME}:${USERNAME} "${VIRTUAL_ENV}" "/workspace"
USER ${USERNAME}
WORKDIR "/workspace"

# Copy package installation instructions and version.txt files
COPY --chown=${USERNAME}:${USERNAME} third_party/stable-baselines3/setup.py ./third_party/stable-baselines3/
COPY --chown=${USERNAME}:${USERNAME} third_party/stable-baselines3/stable_baselines3/version.txt ./third_party/stable-baselines3/stable_baselines3/
COPY --chown=${USERNAME}:${USERNAME} third_party/gym-sokoban/setup.py ./third_party/gym-sokoban/
COPY --chown=${USERNAME}:${USERNAME} third_party/gym-sokoban/README.md ./third_party/gym-sokoban/
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml poetry.lock ./

FROM dependencies as main

# Install content-less packages and their dependencies
RUN mkdir learned_planners experiments \
    && touch learned_planners/__init__.py \
             third_party/stable-baselines3/stable_baselines3/__init__.py \
    && pip install --require-virtualenv --config-settings editable_mode=compat -e '.[dev,launch_jobs]' -e ./third_party/stable-baselines3 -e ./third_party/gym-sokoban \
    && rm -rf "${HOME}/.cache" "./dist" \
    # Run Pyright so its Node.js package gets installed
    && pyright .

ENV ENVPOOL_WHEEL="dist/envpool-0.8.4-cp310-cp310-linux_x86_64.whl"
COPY --from=envpool --chown=${USERNAME}:${USERNAME} "/app/${ENVPOOL_WHEEL}" "./${ENVPOOL_WHEEL}"
RUN pip install "./${ENVPOOL_WHEEL}" && rm -rf "./dist"

# Copy whole repo
COPY --chown=${USERNAME}:${USERNAME} . .

# Set git remote URL to https for all sub-repos
RUN git remote set-url origin "$(git remote get-url origin | sed 's|git@github.com:|https://github.com/|' )" \
    && (cd third_party/stable-baselines3 && git remote set-url origin "$(git remote get-url origin | sed 's|git@github.com:|https://github.com/|' )" )\
    && (cd third_party/gym-sokoban && git remote set-url origin "$(git remote get-url origin | sed 's|git@github.com:|https://github.com/|' )" )\
    && (cd third_party/envpool && git remote set-url origin "$(git remote get-url origin | sed 's|git@github.com:|https://github.com/|' )" )

# Abort if repo is dirty
RUN echo "$(git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && echo "$(cd third_party/stable-baselines3 && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && echo "$(cd third_party/gym-sokoban && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && echo "$(cd third_party/envpool && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && if ! { [ -z "$(git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    && [ -z "$(cd third_party/stable-baselines3 && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    && [ -z "$(cd third_party/gym-sokoban && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    && [ -z "$(cd third_party/envpool && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    ; }; then exit 1; fi

# Default command to run -- may be changed at runtime
CMD ["/bin/bash"]
