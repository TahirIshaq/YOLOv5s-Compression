# YOLOV5s-Compression

## Requirements

```shell
# conda
conda create -n yolov5 python=3.9
conda activate yolov5

# pip
pip install -r requirements.txt
```

## Download data and weight
```shell
# Prepare widerface dataset
python3 prepare_dataset.py

data
├── train
│   └── images
│   └── labels
├── val
│   └── images
│   └── labels
├── wider_face.yaml


# download pre-trained weights
wget -O yolov5s-v5.0.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
wget -O yolov5s-v6.0.pt https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
```

## 1. Base Train
Could not use Yolov5sv6.2 widerface trained model. Need to re-train v6.0 and try.
```shell
# YOLOv5s
python train.py --data data/wider_face.yaml --imgsz 640 --weights yolov5s.pt --cfg models/yolov5s.yaml --epochs 100 --device 0,1 --sync-bn

# [Optional] YOLOv5l-PP-LCNet
python train.py --data data/wider_face.yaml --imgsz 640 --weights yolov5lPP-LC.pt --cfg models/lightModels/yolov5lPP-LC.yaml --epochs 60 --device 0,1 --sync-bn
```

## 2. Sparse train

I. Slim (BN-L1)
```shell
# sparse train using the model trained in step 1
python train.py --data data/wider_face.yaml --imgsz 640 --weights runs/train/exp/weights/best-coco128-mAP05-02293.pt --cfg models/prunModels/yolov5s-pruning.yaml --epochs 100 --device 0,1 --sparse

# prune the sparse trained model in step 2.1.1
python pruneSlim.py --data data/wider_face.yaml --weights runs/2_sparse-coco128-mAP05-035504.pt --cfg models/prunModels/yolov5s-pruning.yaml --path yolov5s-pruned.yaml --global_percent 0.5 --device 0,1

# finetune the pruned model in step 2.1.2
python train.py --data data/wider_face.yaml --imgsz 640 --weights runs/3_sparse-coco128-mAP05-035504-Slimpruned.pt --cfg yolov5s-pruned.yaml --epochs 100 --device 0,1
```

II. EagleEye (un-test)
Eagle eye did not work the model trained in step 1. The following error occurs:
```
Traceback (most recent call last):
  File "/workspace/YOLOv5s-Compression/pruneEagleEye.py", line 123, in <module>
    model.load_state_dict(state_dict, strict=True)       # load strictly
  File "/venv/main/lib/python3.10/site-packages/torch/nn/modules/module.py", line 2584, in load_state_dict
    raise RuntimeError(
RuntimeError: Error(s) in loading state_dict for Model:
	Missing key(s) in state_dict: "model.24.m.0.weight", "model.24.m.0.bias", "model.24.m.1.weight", "model.24.m.1.bias", "model.24.m.2.weight", "model.24.m.2.bias".
```
Skipping eagle eye for now

```shell
# search best sub-net. Prune the sparse trained model in step 2.1.1
python pruneEagleEye.py --data data/wider_face.yaml --weights runs/1_base-coco128-mAP05-02293.pt --cfg models/prunModels/yolov5s-pruning.yaml  --path yolov5s-pruned-eagleeye.yaml --max_iter 100 --remain_ratio 0.5 --delta 0.02

# finetune the pruned model in step 2.2.1
python train.py --data data/wider_face.yaml --imgsz 640 --weights runs/3_base-coco128-mAP05-02293-EagleEyepruned.pt --cfg yolov5s-pruned-eagleeye.yaml --epochs 100 --device 0,1
```


## 5. Quantization PTQ
Copy the pruned model from step 2.1 to its named directory.
```shell
cd yolov5_quant_sample
cp *.pt weights/SlimPrune
cp *.pt weights/EagleEye
```

5.1 export onnx
Export the slimPruned model to ONNX
```shell
python models/export.py --weights weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt --img 640 --batch 1 --device 0,1 
```
This error occured when trying to export the model to ONNX
```
Traceback (most recent call last):
  File "/workspace/YOLOv5s-Compression/models/export.py", line 38, in <module>
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
  File "/workspace/YOLOv5s-Compression/./models/experimental.py", line 96, in attempt_load
    model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
  File "/workspace/YOLOv5s-Compression/./models/yolo.py", line 225, in fuse
    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
  File "/workspace/YOLOv5s-Compression/./utils/torch_utils.py", line 209, in fuse_conv_and_bn
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
RuntimeError: shape '[66, -1]' is invalid for input of size 33984
```

5.2 Build a int8 engine using TensorRT's native PTQ
```shell
rm trt/yolov5s_calibration.cache

python trt/onnx_to_trt.py --model weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.onnx --dtype int8 --batch-size 4 --num-calib-batch 16 --calib-img-dir ../datasets/coco128/images/train2017
```
5.3 Evaluate the accurary of TensorRT inference result.
```shell
# datasets eval (coco128 json label Unfinished)
??? python trt/eval_yolo_trt.py --model ./weights/xxx.trt -l

# image test
python trt/trt_test.py --model weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5-int8-4-16-minmax.trt

python trt/trt_test.py --model weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned-int8-4-16-minmax.trt
```

## 6. Quantization QAT

```shell
# QAT-finetuning
python yolo_quant_flow.py --data data/wider_face.yaml --cfg models/yolov5s.yaml --ckpt-path weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt --hyp data//hyp.qat.yaml --skip-layers
python yolo_quant_flow.py --data data/wider_face.yaml --cfg models/yolov5s.yaml --ckpt-path weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned.pt --hyp data//hyp.qat.yaml --skip-layers

# Build TensorRT engine
python trt/onnx_to_trt.py --model weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5_skip4.onnx --dtype int8 --qat
python trt/onnx_to_trt.py --model weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned_skip4.onnx --dtype int8 --qat

# Evaluate the accuray of TensorRT engine
python trt/trt_test.py --model  weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5_skip4.trt
python trt/trt_test.py --model  weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned_skip4.trt
```

## 7. Export to deploy
```shell
# SlimPrune
python deploy/export_onnx_trt.py --weights runs/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt --device 0,1 --half --simplify

# EagleEye
python deploy/export_onnx_trt.py --weights runs/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned.pt --device 0,1 --half --simplify
```

## Test
1. model size
```shell
Base-coco128-mAP05_0.2293.pt                            14.8M
# SlimPrune
Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt         4.6M
Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.engine     6.7M
Finetune-coco128-mAP05_0.0810-Slimpruned_0.5_skip4.trt  10.1M
# EagleEye
Finetune_coco128-mAP05_0.0860-EagleEyepruned.pt         6.0M
Finetune_coco128-mAP05_0.0860-EagleEyepruned.engine     8.5M
Finetune-coco128-mAP05_0.0860-EagleEyepruned_skip4.trt  10.1M
```
2. detect test
```shell
# Test: original.pt
python detect.py  --weights runs/Base-coco128-mAP05_0.2293.pt
```
Speed: 0.3ms pre-process, 6.0ms inference, 0.4ms NMS per image at shape (1, 3, 640, 640)

```shell
# Test: pruned.pt
python detect.py  --weights runs/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt

python detect.py  --weights runs/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned.pt
```
Speed: 0.3ms pre-process, 6.3ms inference, 0.4ms NMS per image at shape (1, 3, 640, 640)

Speed: 0.3ms pre-process, 6.5ms inference, 0.4ms NMS per image at shape (1, 3, 640, 640)

```shell
# Test: pruned+FP16.engine
python deploy/detect_trt.py --weights runs/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.engine --device 0,1 --half

python deploy/detect_trt.py --weights runs/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned.engine --device 0,1 --half
```
Speed: 0.3ms pre-process, 1.3ms inference, 0.9ms NMS per image at shape (1, 3, 640, 640)

Speed: 0.3ms pre-process, 1.5ms inference, 0.7ms NMS per image at shape (1, 3, 640, 640)

## Result

|  Model   | File Size | Inference per img |
|  ----  | ---- | ----  |
| Base.pt           | 14.8M | 6.0ms |
| SlimPrune.pt      |  **4.6M** | 6.3ms |
| SlimPrune_fp16.engine  |  6.7M | **1.3ms** |
| SlimPrune_int8_PTQ.trt  |  4.9M | 3.5ms |
| SlimPrune_int8_QAT.trt  |  10.1M | 4.1ms |
| EagleEye.pt       |  6.0M | 6.5ms |
| EagleEye_fp16.engine   |  8.5M | **1.5ms** |
| EagleEye_int8_PTQ.trt   |  5.7M | 3.5ms |
| EagleEye_int8_QTA.trt   |  10.1M | 3.9ms |

## Plan
- [X] Base-train
- [X] Prune（SlimPrune、EagleEye）
- [X] Finetune
- [X] FP16 Quantization
- [X] INT8 PTQ
- [X] INT8 QAT
- [ ] Change Backbone(mobilev2/...)
- [ ] Distillation


## Changes made to comply with new library versions
- https://github.com/ultralytics/yolov5/pull/8067/files
- https://github.com/tensorflow/models/issues/11040#issuecomment-1685593292
- https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
- https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load
- https://docs.pytorch.org/docs/stable/amp.html



## Acknowledge
> https://github.com/Gumpest/YOLOv5-Multibackbone-Compression  
> https://github.com/maggiez0138/yolov5_quant_sample  
> https://github.com/Syencil/mobile-yolov5-pruning-distillation
> https://lambda.ai/blog/training-a-yolov5-object-detector-on-lambda-cloud
> https://github.com/LambdaLabsML/examples/blob/main/yolov5/prepare_dataset.py

