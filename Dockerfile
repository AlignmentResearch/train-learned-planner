ARG JAX_DATE

FROM ghcr.io/alignmentresearch/train-learned-planner:614e97a-envpool-devbox AS envpool

RUN --mount=type=cache,target=${HOME}/.cache,uid=${UID},gid=${GID} \
    make bazel-release && cp bazel-bin/*.whl .

FROM ghcr.io/alignmentresearch/flamingo-devbox:jax-${JAX_DATE} AS main
ENV GIT_URL="https://github.com/AlignmentResearch/${APPLICATION_NAME}"
LABEL org.opencontainers.image.authors="Adrià Garriga-Alonso <adria@far.ai>"
LABEL org.opencontainers.image.source=${GIT_URL}

ENV USERNAME=dev
ENV UID=1001
ENV GID=1001
ARG APPLICATION_NAME

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

# Cache Craftax textures
RUN python -c "import craftax.craftax.constants"

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
