### to run docker image
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it greencity-tree-detection
```

### inside of docker image
#### Train & Inference

```
bash run_gpu_train_inference-demo.sh
```
