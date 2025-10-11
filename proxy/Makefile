# Makefile for FastAPI Proxy

# Variables
IMAGE_NAME = ui-verifiers/proxy
TAG = latest
CONTAINER_NAME = proxy-container
PORT = 8000


# Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Run the container
.PHONY: run
run:
	docker run -d --rm -it\
		--name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		$(IMAGE_NAME):$(TAG)

# Push the latest built image to gcp
.PHONY: push
push:
	docker tag ui-verifiers/proxy europe-north2-docker.pkg.dev/my-project-1726641910410/ui-verifiers/proxy
	docker push europe-north2-docker.pkg.dev/my-project-1726641910410/ui-verifiers/proxy


# Deploy the proxy to kubernetes
.PHONY: deploy
deploy:
	kubectl apply -f deploy-proxy.yaml