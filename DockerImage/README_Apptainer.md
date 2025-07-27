### To run Apptainer image

First, build the Apptainer container (note that building requires root privileges):

```
sudo apptainer build greencity-tree-detection.sif Apptainer.def
```

Then, run the container interactively with NVIDIA GPU support and mount your data directory:

```
apptainer shell --nv greencity-tree-detection.sif
```

### Inside the Apptainer container

#### Train & Inference

Once inside the container, navigate to the mounted data directory and run the inference commands:

```
bash run_gpu_train_inference-demo.sh
```