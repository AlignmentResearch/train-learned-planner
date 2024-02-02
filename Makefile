APPLICATION_NAME ?= lp-cleanba
APPLICATION_URL ?= ghcr.io/alignmentresearch/${APPLICATION_NAME}
DOCKERFILE ?= Dockerfile
SHELL ?= /bin/bash

export APPLICATION_NAME
export APPLICATION_URL
export DOCKERFILE

# By default, the base tag is the short git hash
BASE_TAG ?= $(shell git rev-parse --short HEAD)
COMMIT_FULL ?= $(shell git rev-parse HEAD)
BRANCH_NAME ?= $(shell git branch --show-current)

# Release tag or latest if not a release
RELEASE_TAG ?= latest
DEFAULT_TARGET ?= main

.PHONY: default
default: build-${BASE_TAG}-${DEFAULT_TARGET}

# Re-usable function
.PHONY: build
build:
	docker pull "${APPLICATION_URL}:${TAG}-${TARGET}" || true
	docker build --platform "linux/amd64" \
		--tag "${APPLICATION_URL}:${TAG}-${TARGET}" \
		--build-arg "APPLICATION_NAME=${APPLICATION_NAME}" \
		--target ${TARGET} \
		-f "${DOCKERFILE}" .

.PHONY: build-%-main
build-%-main: ${DOCKERFILE}
	$(MAKE) build "TAG=$*" TARGET=main

.PHONY: build-%-envpool
build-%-envpool: ${DOCKERFILE}
	$(MAKE) build "TAG=$*" TARGET=envpool

.PHONY: build-%-dependencies
build-%-dependencies: ${DOCKERFILE}
	$(MAKE) build "TAG=$*" TARGET=dependencies

.PHONY: build-%-jax
build-%-jax: ${DOCKERFILE}
	$(MAKE) build "TAG=$*" TARGET=jax

.PHONY: push-%
push-%: build-%
	docker push "${APPLICATION_URL}:$*"

release: release-${BASE_TAG}-${DEFAULT_TARGET}

.PHONY: release-%-main
release-%-main: push-%-main
	docker tag "${APPLICATION_URL}:$*-main" ${APPLICATION_URL}:${RELEASE_TAG}-main
	docker push "${APPLICATION_URL}:${RELEASE_TAG}-main"

.PHONY: devbox devbox-%
devbox: devbox-${DEFAULT_TARGET}
devbox-%:
	git push
	python -c "print(open('k8s/devbox.yaml').read().format(NAME='devbox-$*', IMAGE='${APPLICATION_URL}:latest-$*', COMMIT_FULL='${COMMIT_FULL}', CPU=14, MEMORY='60G', GPU=0, USER_ID=1001, GROUP_ID=1001))" | kubectl create -f -
