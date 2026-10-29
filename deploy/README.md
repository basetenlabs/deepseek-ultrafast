# Deployment

## Build the image

```sh
docker build -f deploy/Dockerfile --build-arg DSUF_SM=sm_103a -t deepseek-ultrafast .   # GB300, B300
docker build -f deploy/Dockerfile -t deepseek-ultrafast .                                  # B200
```

## Run with Docker

```sh
docker run -d --network host redis:7 redis-server --bind 127.0.0.1 --save '' --appendonly no
docker run -d --gpus all --network host --device /dev/infiniband --ulimit memlock=-1 \
    -v "$PWD/models:/models:ro" deepseek-ultrafast
```

The image serves on GPUs 0-3 by default. On an 8-GPU B300 machine, serve TP8 on all eight:

```sh
docker run -d --gpus all --network host --device /dev/infiniband --ulimit memlock=-1 \
    -v "$PWD/models:/models:ro" deepseek-ultrafast --gpus 0,1,2,3,4,5,6,7 --port 8000
```

## Run on Kubernetes

Set the image and the `deepseek-ultrafast-models` volume in `kubernetes.yaml`, then:

```sh
kubectl apply -f deploy/kubernetes.yaml
```

For TP8 on 8-GPU B300 nodes, set the container's `--gpus 0,1,2,3,4,5,6,7` and `nvidia.com/gpu: 8`.

## Configuration

- `DSUF_REDIS_URL` (default `redis://127.0.0.1:6379`): the Redis shared by replicas
- `DSUF_KV_HOST`, `DSUF_KV_PORT` (default port `19500`): the address peers reach for KV transfer
- `serve` flags: `--gpus`, `--port`, `--max-seq`, `--chunk`, `--model-name`, `--ckpt`
