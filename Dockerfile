ARG JAX_DATE

FROM ghcr.io/nvidia/jax:base-${JAX_DATE} AS envpool-environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y golang-1.21 git \
    # Linters
      clang-format clang-tidy \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ubuntu
ENV HOME=/home/ubuntu
ENV PATH=/usr/lib/go-1.21/bin:${HOME}/go/bin:$PATH
ENV UID=1000
ENV GID=1000
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} go install github.com/bazelbuild/bazelisk@v1.19.0 && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} go install github.com/bazelbuild/buildtools/buildifier@v0.0.0-20231115204819-d4c9dccdfbb1
# Install Go linting tools
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} go install github.com/google/addlicense@v1.1.1

ENV USE_BAZEL_VERSION=8.1.0
RUN bazel version

WORKDIR /app
# Copy the whole repository
COPY --chown=ubuntu:ubuntu third_party/envpool .

# Install python-based linting dependencies
COPY --chown=ubuntu:ubuntu \
    third_party/envpool/third_party/pip_requirements/requirements-devtools.txt \
    third_party/pip_requirements/requirements-devtools.txt
RUN --mount=type=cache,target=${HOME}/.cache,uid=1000,gid=1000 pip install -r third_party/pip_requirements/requirements-devtools.txt

# Deal with the fact that envpool is a submodule and has no .git directory
RUN rm .git
# Copy the .git repository for this submodule
COPY --chown=ubuntu:ubuntu .git/modules/envpool ./.git
# Remove config line stating that the worktree for this repo is elsewhere
RUN sed -e 's/^.*worktree =.*$//' .git/config > .git/config.new && mv .git/config.new .git/config

# Abort if repo is dirty
RUN echo "$(git status --porcelain --ignored=traditional)" \
    && if ! { [ -z "$(git status --porcelain --ignored=traditional)" ] \
    ; }; then exit 1; fi

FROM envpool-environment AS envpool
RUN --mount=type=cache,target=${HOME}/.cache,uid=1000,gid=1000 make bazel-release && cp bazel-bin/*.whl .

FROM ghcr.io/nvidia/jax:jax-${JAX_DATE} AS main-pre-pip

ARG APPLICATION_NAME
ARG UID=1001
ARG GID=1001
ARG USERNAME=dev

ENV GIT_URL="https://github.com/AlignmentResearch/${APPLICATION_NAME}"

ENV DEBIAN_FRONTEND=noninteractive
MAINTAINER Adrià Garriga-Alonso <adria@far.ai>
LABEL org.opencontainers.image.source=${GIT_URL}

RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
    # essential for running.
    git git-lfs tini python3-dev python3-venv \
    # devbox niceties
    curl vim tmux less sudo \
    # CircleCI
    ssh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tini: reaps zombie processes and forwards signals
ENTRYPOINT ["/usr/bin/tini", "--"]

# Devbox niceties
WORKDIR "/devbox-niceties"
## the Unison file synchronizer
RUN curl -OL https://github.com/bcpierce00/unison/releases/download/v2.53.4/unison-2.53.4-ubuntu-x86_64-static.tar.gz \
    && tar xf unison-2.53.4-ubuntu-x86_64-static.tar.gz \
    && mv bin/unison bin/unison-fsmonitor /usr/local/bin/ \
    && rm -rf /devbox-niceties \
## Terminfo for the Alacritty terminal
    && curl -L https://raw.githubusercontent.com/alacritty/alacritty/master/extra/alacritty.info | tic -x /dev/stdin

# Simulate virtualenv activation
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python3 -m venv "${VIRTUAL_ENV}" --system-site-packages \
    && addgroup --gid ${GID} ${USERNAME} \
    && adduser --uid ${UID} --gid ${GID} --disabled-password --gecos '' ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p "/workspace" \
    && chown -R ${USERNAME}:${USERNAME} "${VIRTUAL_ENV}" "/workspace"
USER ${USERNAME}
WORKDIR "/workspace"

# Get a pip modern enough that can resolve farconf
RUN pip install "pip ==24.0" && rm -rf "${HOME}/.cache"

FROM main-pre-pip AS main-pip-tools
RUN pip install "pip-tools ~=7.4.1"

FROM main-pre-pip AS main
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} pip install uv
COPY --chown=${USERNAME}:${USERNAME} requirements.txt ./
# Install all dependencies, which should be explicit in `requirements.txt`
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} \
    uv pip sync requirements.txt \
    # Run Pyright so its Node.js package gets installed
    && pyright .

# Install Envpool
ENV ENVPOOL_WHEEL="envpool-0.9.0-cp312-cp312-linux_x86_64.whl"
COPY --from=envpool --chown=${USERNAME}:${USERNAME} "/app/${ENVPOOL_WHEEL}" "${ENVPOOL_WHEEL}"
RUN uv pip install "${ENVPOOL_WHEEL}" && rm "${ENVPOOL_WHEEL}"

# Copy whole repo
COPY --chown=${USERNAME}:${USERNAME} . .
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} \
    uv pip install --no-deps -e . -e ./third_party/gym-sokoban/

# Set git remote URL to https for all sub-repos
RUN git remote set-url origin "$(git remote get-url origin | sed 's|git@github.com:|https://github.com/|' )" \
    && (cd third_party/envpool && git remote set-url origin "$(git remote get-url origin | sed 's|git@github.com:|https://github.com/|' )" )

# Abort if repo is dirty
RUN rm NVIDIA_Deep_Learning_Container_License.pdf \
    && echo "$(git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && echo "$(cd third_party/envpool && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && echo "$(cd third_party/gym-sokoban && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && if ! { [ -z "$(git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    && [ -z "$(cd third_party/envpool && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    && [ -z "$(cd third_party/gym-sokoban && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    ; }; then exit 1; fi


FROM main AS atari
RUN uv pip uninstall -y envpool && uv pip install envpool && rm -rf "${HOME}/.cache"
