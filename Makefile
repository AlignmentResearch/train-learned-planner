APPLICATION_NAME ?= lp-cleanba
APPLICATION_URL ?= ghcr.io/alignmentresearch/${APPLICATION_NAME}
DOCKERFILE ?= Dockerfile

export APPLICATION_NAME
export APPLICATION_URL
export DOCKERFILE

COMMIT_SHORT ?= $(shell git rev-parse --short HEAD)
COMMIT_FULL ?= $(shell git rev-parse HEAD)
BRANCH_NAME ?= $(shell git branch --show-current)

# Release tag or latest if not a release
RELEASE_TAG ?= latest
TARGET ?= main
TAG ?= ${COMMIT_SHORT}

default: release

# Re-usable function
.PHONY: build
build: ${DOCKERFILE}
	docker pull "${APPLICATION_URL}:${TAG}-${TARGET}" || true
	docker build --platform "linux/amd64" \
		--tag "${APPLICATION_URL}:${TAG}-${TARGET}" \
		--build-arg "APPLICATION_NAME=${APPLICATION_NAME}" \
		--target ${TARGET} \
		-f "${DOCKERFILE}" .

.PHONY: push
push: build
	docker push "${APPLICATION_URL}:${TAG}-${TARGET}"

.PHONY: release
release: build
	docker tag "${APPLICATION_URL}:${TAG}-${TARGET}" "${APPLICATION_URL}:${RELEASE_TAG}-${TARGET}"
	docker push "${APPLICATION_URL}:${RELEASE_TAG}-${TARGET}"

DEVBOX_UID ?= 1001
CPU ?= 14
MEMORY ?= 60G
GPU ?= 0

DEVBOX_NAME ?= cleanba-devbox

.PHONY: devbox devbox-main cpu-devbox
create-devbox:
	git push
	python -c "print(open('k8s/devbox.yaml').read().format(NAME='${DEVBOX_NAME}', IMAGE='${APPLICATION_URL}:${RELEASE_TAG}-${TARGET}', COMMIT_FULL='${COMMIT_FULL}', CPU='${CPU}', MEMORY='${MEMORY}', GPU='${GPU}', USER_ID=${DEVBOX_UID}, GROUP_ID=${DEVBOX_UID}))" | kubectl create -f -

.PHONY: cpu-devbox
cpu-devbox: GPU=0
cpu-devbox: create-devbox

.PHONY: devbox
devbox: GPU=1
devbox: create-devbox
