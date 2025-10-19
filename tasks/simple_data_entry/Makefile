# Build image
.PHONY: build
build:
	docker build -t ui-verifiers/simple-data-entry .

# Run locally 
.PHONY: run
run:
	docker run --rm -it -p 5900:5900 -p 8000:8000 --name simple-data-entry ui-verifiers/simple-data-entry

# Push the latest built image to gcp
.PHONY: push
push:
	docker tag ui-verifiers/simple-data-entry europe-north2-docker.pkg.dev/my-project-1726641910410/ui-verifiers/simple-data-entry
	docker push europe-north2-docker.pkg.dev/my-project-1726641910410/ui-verifiers/simple-data-entry
