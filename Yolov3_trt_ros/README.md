# YOLOv3-pytorch

single stage object detection Yolov3.

----------------------------
### Linux

#### Use docker

[Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) docker image

1. docker pull
```{r, engine='bash', count_lines}
docker pull nvcr.io/nvidia/pytorch:21.12-py3
```
2. docker run
```{r, engine='bash', count_lines}
docker run \
--name yolov3_tstl \
--gpus all \
-p 6006:8888 \
-it \
-v ${PWD}/DockerFile/yolov3-pytorch:/yolov3-pytorch \
-v /home/bert/Downloads/datasets/tstl/tstl_dataset:/yolov3-pytorch/datasets \
nvcr.io/nvidia/pytorch:21.12-py3 /bin/bash
```
   - check `nvidia-smi`
   - check `nvcc --version`
3. docker exec
```{r, engine='bash', count_lines}
docker start yolov3_tstl
docker exec -it yolov3_tstl /bin/bash
```
-------------------

## Run

training,

```{r, engine='bash', count_lines}
(single gpu) 
python main.py \
--mode train --cfg ./yolov3.cfg \
--gpus 0 \
--checkpoint ${saved_checkpoint_path}

(multi gpu) 
python main.py \
--mode train --cfg ./yolov3.cfg \
--gpus 0 1 2 3 \
--checkpoint ${saved_checkpoint_path}
```

evaluate,

```{r, engine='bash', count_lines}
python main.py \
--mode eval \
--cfg ./yolov3.cfg \
--gpus 0 \
--checkpoint ${saved_checkpoint_path}
```

test,

```{r, engine='bash', count_lines}
python main.py \
--mode demo \
--cfg ./yolov3.cfg \
--gpus 0 \
--checkpoint ${saved_checkpoint_path}
```

converting torch to onnx,

target tensorrt version > 7
```{r, engine='bash', count_lines}
python main.py \
--mode onnx \
--cfg ./cfg/yolov3.cfg \
--gpus 0 \
--checkpoint ${saved_checkpoint_path}
```

target tensorrt version is 5.x

1. **ONNX_EXPORT = True** in 'model/yolov3.py'
   
   tensorrt(v5.x) is not support upsample scale factor, so you have to change upsample layer not using scale factor.

```{r, engine='bash', count_lines}
python main.py \
--mode onnx \
--cfg ./cfg/yolov3.cfg \
--gpus 0 \
--checkpoint ${saved_checkpoint_path}
```

### option

--mode : train/eval/demo.

--cfg : the path of model.cfg.

--gpu : if you use GPU, set 1. If you use CPU, set 0.

--checkpoint (optional) : the path of saved model checkpoint. Use it when you want to load the previous train, or you want to test(evaluate) the model.

--pretrained (optional) : the path of darknet pretrained weights. Use it when you want to fine-tuning the model.



## Visualize training graph

Using Tensorboard,

```{r, engine='bash', count_lines}
tensorboard --logdir=./output --port 8888
```

-------------------------

# Reference

[YOLOv3 paper](https://arxiv.org/abs/1804.02767)

[KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)


# error

- if libgl.so error when cv2
```
apt-get update
apt-get install libgl1-mesa-glx
```