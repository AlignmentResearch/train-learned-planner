ARG JAX_DATE

FROM ghcr.io/nvidia/jax:base-${JAX_DATE} as envpool-environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y golang-1.18 git \
    # Linters
      clang-format clang-tidy \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/lib/go-1.18/bin:/root/go/bin:$PATH
RUN go install github.com/bazelbuild/bazelisk@v1.19.0 && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@v0.0.0-20231115204819-d4c9dccdfbb1
# Install Go linting tools
RUN go install github.com/google/addlicense@v1.1.1

ENV USE_BAZEL_VERSION=6.4.0
RUN bazel version

WORKDIR /app
# Install python-based linting dependencies
COPY third_party/envpool/third_party/pip_requirements/requirements-devtools.txt \
    third_party/pip_requirements/requirements-devtools.txt
RUN pip install -r third_party/pip_requirements/requirements-devtools.txt

# Copy the whole repository
COPY third_party/envpool .

# Deal with the fact that envpool is a submodule and has no .git directory
RUN rm .git
# Copy the .git repository for this submodule
COPY .git/modules/envpool ./.git
# Remove config line stating that the worktree for this repo is elsewhere
RUN sed -e 's/^.*worktree =.*$//' .git/config > .git/config.new && mv .git/config.new .git/config

# Abort if repo is dirty
RUN echo "$(git status --porcelain --ignored=traditional)" \
    && if ! { [ -z "$(git status --porcelain --ignored=traditional)" ] \
    ; }; then exit 1; fi

FROM envpool-environment as envpool
RUN make bazel-release

FROM ghcr.io/nvidia/jax:jax-${JAX_DATE} as main-pre-pip

ARG APPLICATION_NAME
ARG USERID=1001
ARG GROUPID=1001
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
    && addgroup --gid ${GROUPID} ${USERNAME} \
    && adduser --uid ${USERID} --gid ${GROUPID} --disabled-password --gecos '' ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p "/workspace" \
    && chown -R ${USERNAME}:${USERNAME} "${VIRTUAL_ENV}" "/workspace"
USER ${USERNAME}
WORKDIR "/workspace"

# Get a pip modern enough that can resolve farconf
RUN pip install "pip ==24.0" && rm -rf "${HOME}/.cache"

FROM main-pre-pip as main-pip-tools
RUN pip install "pip-tools ~=7.4.1"

FROM main-pre-pip as main
COPY --chown=${USERNAME}:${USERNAME} requirements.txt ./
# Install all dependencies, which should be explicit in `requirements.txt`
RUN pip install --no-deps -r requirements.txt \
    && rm -rf "${HOME}/.cache" \
    # Run Pyright so its Node.js package gets installed
    && pyright .

# Install Envpool
ENV ENVPOOL_WHEEL="dist/envpool-0.8.4-cp310-cp310-linux_x86_64.whl"
COPY --from=envpool --chown=${USERNAME}:${USERNAME} "/app/${ENVPOOL_WHEEL}" "./${ENVPOOL_WHEEL}"
RUN pip install "./${ENVPOOL_WHEEL}" && rm -rf "./dist"

# Copy whole repo
COPY --chown=${USERNAME}:${USERNAME} . .
RUN pip install --no-deps -e . -e ./third_party/gym-sokoban/

# Set git remote URL to https for all sub-repos
RUN git remote set-url origin "$(git remote get-url origin | sed 's|git@github.com:|https://github.com/|' )" \
    && (cd third_party/envpool && git remote set-url origin "$(git remote get-url origin | sed 's|git@github.com:|https://github.com/|' )" )

# Abort if repo is dirty
RUN echo "$(git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && echo "$(cd third_party/envpool && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && echo "$(cd third_party/gym-sokoban && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" \
    && if ! { [ -z "$(git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    && [ -z "$(cd third_party/envpool && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    && [ -z "$(cd third_party/gym-sokoban && git status --porcelain --ignored=traditional | grep -v '.egg-info/$')" ] \
    ; }; then exit 1; fi


FROM main as atari
RUN pip uninstall -y envpool && pip install envpool && rm -rf "${HOME}/.cache"
