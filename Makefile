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

.PHONY: default
default: build-${BASE_TAG}-dependencies

# Re-usable function
.PHONY: build
build:
	docker pull ${APPLICATION_URL}:${TAG} || true
	docker build --platform "linux/amd64" \
		--tag "${APPLICATION_URL}:${TAG}" \
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

.PHONY: push-%
push-%: build-%
	docker push "${APPLICATION_URL}:$*"

.PHONY: release-%
release-%: push-%
	docker tag ${APPLICATION_URL}:$* ${APPLICATION_URL}:${RELEASE_TAG}
	docker push ${APPLICATION_URL}:${RELEASE_TAG}
