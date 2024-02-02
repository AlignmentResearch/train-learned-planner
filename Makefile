APPLICATION_NAME ?= lp-cleanba
APPLICATION_URL ?= ghcr.io/alignmentresearch/${APPLICATION_NAME}
DOCKERFILE ?= Dockerfile

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

.PHONY: release-%
release-%: push-%
	docker tag ${APPLICATION_URL}:$* ${APPLICATION_URL}:${RELEASE_TAG}
	docker push ${APPLICATION_URL}:${RELEASE_TAG}
