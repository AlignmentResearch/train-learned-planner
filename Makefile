APPLICATION_NAME ?= lp-cleanba
APPLICATION_URL ?= ghcr.io/alignmentresearch/${APPLICATION_NAME}
DOCKERFILE ?= Dockerfile

export APPLICATION_NAME
export APPLICATION_URL
export DOCKERFILE

COMMIT_FULL ?= $(shell git rev-parse HEAD)
BRANCH_NAME ?= $(shell git branch --show-current)

default: release/main

# Section 1: Build Dockerfiles
## Release tag or latest if not a release
RELEASE_PREFIX ?= latest
BUILD_PREFIX ?= $(shell git rev-parse --short HEAD)

.build/with-reqs/${BUILD_PREFIX}/%: requirements.txt
	mkdir -p .build/with-reqs/${BUILD_PREFIX}
	docker pull "${APPLICATION_URL}:${BUILD_PREFIX}-$*" || true
	docker build --platform "linux/amd64" \
		--tag "${APPLICATION_URL}:${BUILD_PREFIX}-$*" \
		--build-arg "APPLICATION_NAME=${APPLICATION_NAME}" \
		--target "$*" \
		-f "${DOCKERFILE}" .
	touch ".build/with-reqs/${BUILD_PREFIX}/$*"

# NOTE: --extra=extra is for stable-baselines3 testing.
#
# We use RELEASE_PREFIX as the image so we don't have to re-build it constantly. Once we have bootstrapped
# `requirements.txt`, we can push the image with `make release/main-pip-tools`
requirements.txt.new: pyproject.toml ${DOCKERFILE}
	docker run -v "${HOME}/.cache:/home/dev/.cache" -v "$(shell pwd):/workspace" "${APPLICATION_URL}:${RELEASE_PREFIX}-main-pip-tools" \
	pip-compile --verbose -o requirements.txt.new \
		--extra=dev --extra=launch_jobs pyproject.toml

requirements.txt: requirements.txt.new
	sed -E "s/^(jax==.*|jaxlib==.*|nvidia-.*|torchvision==.*|torch==.*|triton==.*)$$/# DISABLED \\1/g" requirements.txt.new > requirements.txt

.PHONY: local-install
local-install: requirements.txt
	pip install --no-deps -r requirements.txt
	pip install --config-settings editable_mode=compat \
        -e ".[dev-local]" -e ./third_party/stable-baselines3 -e ./third_party/gym-sokoban -e ./third_party/farconf


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

# Section 2: Make Devboxes and local devboxes (with Docker)
DEVBOX_UID ?= 1001
CPU ?= 1
MEMORY ?= 60G
GPU ?= 0

DEVBOX_NAME ?= cleanba-devbox

.PHONY: devbox devbox/%
devbox/%:
	git push
	python -c "print(open('k8s/devbox.yaml').read().format(NAME='${DEVBOX_NAME}', IMAGE='${APPLICATION_URL}:${RELEASE_PREFIX}-$*', COMMIT_FULL='${COMMIT_FULL}', CPU='${CPU}', MEMORY='${MEMORY}', GPU='${GPU}', USER_ID=${DEVBOX_UID}, GROUP_ID=${DEVBOX_UID}))" | kubectl create -f -
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
	ruff --fix .

format:
	ruff format .

typecheck:
	pyright .


mactest:
	pytest -k 'not test_environment_basics[cfg2]'
