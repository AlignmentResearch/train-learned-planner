ARG JAX_DATE

FROM ghcr.io/alignmentresearch/flamingo-devbox:jax-${JAX_DATE} AS envpool-devbox
ENV DEBIAN_FRONTEND=noninteractive
RUN sudo apt-get update \
    && sudo apt-get install -y \
    # Linters
    clang-format clang-tidy \
    # Cmake dependencies, for building CMake to build Envpool
    openssl libssl-dev \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

# Install Go 1.24.0 from official source
RUN curl -OL https://go.dev/dl/go1.24.0.linux-amd64.tar.gz \
    && sudo rm -rf /usr/local/go \
    && sudo tar -C /usr/local -xzf go1.24.0.linux-amd64.tar.gz \
    && rm go1.24.0.linux-amd64.tar.gz


# Install bazel-remote
RUN curl -OL 'https://github.com/buchgr/bazel-remote/releases/download/v2.5.0/bazel-remote-2.5.0-linux-x86_64' \
    && chmod +x bazel-remote-2.5.0-linux-x86_64 \
    && sudo mv bazel-remote-2.5.0-linux-x86_64 /usr/local/bin/bazel-remote
ENV USERNAME=dev
ENV UID=1001
ENV GID=1001
USER ${USERNAME}
ENV HOME=/home/${USERNAME}
ENV PATH=/usr/local/go/bin:${HOME}/go/bin:$PATH

RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} go install github.com/bazelbuild/bazelisk@v1.25.0 && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} go install github.com/bazelbuild/buildtools/buildifier@v0.0.0-20231115204819-d4c9dccdfbb1
# Install Go linting tools
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} go install github.com/google/addlicense@v1.1.1

ENV USE_BAZEL_VERSION=8.1.0
RUN bazel version

WORKDIR /app
# Copy the whole repository
COPY --chown=${USERNAME}:${USERNAME} third_party/envpool .

# Install python-based linting dependencies
COPY --chown=${USERNAME}:${USERNAME} \
    third_party/envpool/third_party/pip_requirements/requirements-devtools.txt \
    third_party/pip_requirements/requirements-devtools.txt
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} pip install -r third_party/pip_requirements/requirements-devtools.txt
ENV PATH="${HOME}/.local/bin:${PATH}"

# Deal with the fact that envpool is a submodule and has no .git directory
RUN rm .git
# Copy the .git repository for this submodule
COPY --chown=${USERNAME}:${USERNAME} .git/modules/envpool ./.git
# Remove config line stating that the worktree for this repo is elsewhere
RUN sed -e 's/^.*worktree =.*$//' .git/config > .git/config.new && mv .git/config.new .git/config

FROM envpool-devbox AS envpool

RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} \
    make bazel-release && cp bazel-bin/*.whl .

FROM ghcr.io/alignmentresearch/flamingo-devbox:jax-${JAX_DATE} AS main-pre-pip

ARG APPLICATION_NAME

ENV GIT_URL="https://github.com/AlignmentResearch/${APPLICATION_NAME}"

LABEL org.opencontainers.image.authors="Adrià Garriga-Alonso <adria@far.ai>"
LABEL org.opencontainers.image.source=${GIT_URL}

# Get a pip modern enough that can resolve farconf
RUN pip install "pip ==24.0" && rm -rf "${HOME}/.cache"

FROM main-pre-pip AS main-pip-tools
RUN pip install "pip-tools ~=7.4.1"

FROM main-pre-pip AS main
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} pip install uv
COPY --chown=${USERNAME}:${USERNAME} requirements.txt ./
# Install all dependencies, which should be explicit in `requirements.txt`
RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} \
    uv pip install --no-deps -r requirements.txt \
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
    # && echo "$(cd third_party/envpool && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && echo "$(cd third_party/gym-sokoban && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && if ! { [ -z "$(git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    # && [ -z "$(cd third_party/envpool && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    && [ -z "$(cd third_party/gym-sokoban && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    ; }; then exit 1; fi
