# BISONN plugin Makefile
# Usage:
#   make build   — build the plugin image (podman or docker)
#   make push    — push to a registry (set REGISTRY + TAG)
#   make test    — syntax/import check (no GPU needed)
#   make run     — one-shot test with default args (needs pluginctl + GPU)

IMAGE ?= bisonn
TAG ?= 0.1.0
REGISTRY ?= localhost
FULL_REF = $(REGISTRY)/$(IMAGE):$(TAG)
PY ?= python3

.PHONY: all build push test run clean

all: build

build:
	podman build -t $(FULL_REF) . || docker build -t $(FULL_REF) .

push:
	podman push $(FULL_REF) || docker push $(FULL_REF)

test:
	$(PY) -m py_compile app.py
	$(PY) -c "import yaml; yaml.safe_load(open('sage.yaml'))" && echo "sage.yaml OK"
	@test -f models/svm.joblib && echo "svm.joblib OK"
	@test -f models/label_names.json && echo "label_names.json OK"

run:
	sudo pluginctl run --name bisonn-test --selector zone=core $(FULL_REF) -- --image-dir /app/test-images --continuous N

clean:
	rm -rf __pycache__ *.pyc
