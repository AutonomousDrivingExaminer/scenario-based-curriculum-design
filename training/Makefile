.PHONY:

train-base-image:
	docker build \
	-t adex/training-base \
	-f docker/training.base.dockerfile ..

train-server-image: train-base-image
	docker build \
	-t adex/training-server \
	-f docker/training.server.dockerfile .

all: train-base-image train-server-image