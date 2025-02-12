APPLICATION_NAME ?= train-learned-planner
APPLICATION_URL ?= ghcr.io/alignmentresearch/${APPLICATION_NAME}
DOCKERFILE ?= Dockerfile

export APPLICATION_NAME
export APPLICATION_URL
export DOCKERFILE

COMMIT_HASH ?= $(shell git rev-parse HEAD)
BRANCH_NAME ?= $(shell git branch --show-current)
JAX_DATE=2024-04-08

default: release/main

# Section 1: Build Dockerfiles
## Release tag or latest if not a release
RELEASE_PREFIX ?= latest
BUILD_PREFIX ?= $(shell git rev-parse --short HEAD)

# NOTE: to bootstrap `pip-tools` image, comment out the `requirements.txt` dependency here and run `make release/main-pip-tools`
.build/with-reqs/${BUILD_PREFIX}/%: requirements.txt
	mkdir -p .build/with-reqs/${BUILD_PREFIX}
	docker pull "${APPLICATION_URL}:${BUILD_PREFIX}-$*" || true
	docker build --platform "linux/amd64" \
		--tag "${APPLICATION_URL}:${BUILD_PREFIX}-$*" \
		--build-arg "APPLICATION_NAME=${APPLICATION_NAME}" \
		--build-arg "JAX_DATE=${JAX_DATE}" \
		--target "$*" \
		-f "${DOCKERFILE}" .
	touch ".build/with-reqs/${BUILD_PREFIX}/$*"

# NOTE: --extra=extra is for stable-baselines3 testing.
requirements.txt.new: pyproject.toml ${DOCKERFILE}
	docker run -v "${HOME}/.cache:/home/dev/.cache" -v "$(shell pwd):/workspace" "ghcr.io/nvidia/jax:base-${JAX_DATE}" \
    bash -c "pip install pip-tools \
		&& cd /workspace \
		&& pip-compile --verbose -o requirements.txt.new --extra=dev --extra=launch_jobs pyproject.toml"

# To bootstrap `requirements.txt`, comment out this target
requirements.txt: requirements.txt.new
	sed -E "s/^(jax==.*|jaxlib==.*|nvidia-.*|torchvision==.*|torch==.*|triton==.*)$$/# DISABLED \\1/g" requirements.txt.new > requirements.txt

.PHONY: local-install
local-install: requirements.txt
	pip install --no-deps -r requirements.txt
	pip install --config-settings editable_mode=compat -e ".[dev-local]" -e ./third_party/gym-sokoban
	pip install https://github.com/AlignmentResearch/envpool/releases/download/v0.1.0/envpool-0.8.4-cp310-cp310-linux_x86_64.whl


.PHONY: build build/%
build/%: .build/with-reqs/${BUILD_PREFIX}/%
	true
build: build/main
	true

.PHONY: push push/%
push/%: .build/with-reqs/${BUILD_PREFIX}/%
	docker push "${APPLICATION_URL}:${BUILD_PREFIX}-$*"
push: push/main

.PHONY: release release/%
release/%: push/%
	docker tag "${APPLICATION_URL}:${BUILD_PREFIX}-$*" "${APPLICATION_URL}:${RELEASE_PREFIX}-$*"
	docker push "${APPLICATION_URL}:${RELEASE_PREFIX}-$*"
release: release/main

.PHONY: release-remote release-remote/%
release-remote/%:
	git push
	python -c "print(open('k8s/kaniko-build.yaml').read().format(APPLICATION_NAME='${APPLICATION_NAME}', JAX_DATE='${JAX_DATE}', BUILD_TAG='${BUILD_PREFIX}-$*', RELEASE_TAG='${RELEASE_PREFIX}-$*', COMMIT_HASH='${COMMIT_HASH}', BRANCH_NAME='${BRANCH_NAME}'))" | kubectl create -f -
release-remote: release-remote/main

# Section 2: Make Devboxes and local devboxes (with Docker)
DEVBOX_UID ?= 1001
CPU ?= 1
MEMORY ?= 60G
SHM_SIZE ?= 20G
GPU ?= 0

DEVBOX_NAME ?= cleanba-devbox

.PHONY: devbox devbox/%
devbox/%:
	git push
	python -c "print(open('k8s/devbox.yaml').read().format(NAME='${DEVBOX_NAME}', IMAGE='${APPLICATION_URL}:${RELEASE_PREFIX}-$*', COMMIT_HASH='${COMMIT_HASH}', CPU='${CPU}', MEMORY='${MEMORY}', SHM_SIZE='${SHM_SIZE}', GPU='${GPU}', USER_ID=${DEVBOX_UID}, GROUP_ID=${DEVBOX_UID}))" | kubectl create -f -
devbox: devbox/main

.PHONY: cuda-devbox cuda-devbox/%
cuda-devbox/%: GPU=1
cuda-devbox/%: devbox/%
	true  # Do nothing, the body has to have something otherwise Makefile complains

cuda-devbox: cuda-devbox/main

.PHONY: envpool-devbox
envpool-devbox: devbox/envpool-ci


.PHONY: docker docker/%
docker/%:
	docker run -v "$(shell pwd):/workspace" -it "${APPLICATION_URL}:${RELEASE_PREFIX}-$*" /bin/bash
docker: docker/main

.PHONY: envpool-docker envpool-docker/%
envpool-docker/%:
	docker run -v "$(shell pwd)/third_party/envpool:/app" -it "${APPLICATION_URL}:${RELEASE_PREFIX}-$*" /bin/bash
envpool-docker: envpool-docker/envpool-ci

# Section 3: project commands

.PHONY: lint format typecheck mactest

lint:
	ruff check --fix .

lint-check:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	pyright .


PYTEST_ARGS ?=
mactest:
	pytest ${PYTEST_ARGS} -m 'not envpool'
