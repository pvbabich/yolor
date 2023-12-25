# YOLOR
implementation of paper - [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206)

## Testing

```
python test.py --img 1280 --conf 0.001 --iou 0.65 --batch 32 --device 0 --data data/coco.yaml --weights yolor-p6.pt --name yolor-p6_val
```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.52471
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.70569
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.57419
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.37395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.57292
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.65187
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38975
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.65382
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.71349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.58020
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.75261
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.82403
```

## Training

Single GPU training:

```
python train.py --batch-size 4 --img 1280 1280 --data path/to/your/data.yaml --cfg models/yolor-w6.yaml --weights yolor-w6.pt --device 0 --hyp data/your.hyp.file.yaml --epochs 300`
python train.py --batch-size 8 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6.yaml --weights '' --device 0 --name yolor-p6 --hyp hyp.scratch.1280.yaml --epochs 300
```

Multiple GPU training:

```
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --batch-size 16 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6.yaml --weights '' --sync-bn --device 0,1 --name yolor-p6 --hyp hyp.scratch.1280.yaml --epochs 300
```

Training schedule in the paper:

```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6.yaml --weights '' --sync-bn --device 0,1,2,3,4,5,6,7 --name yolor-p6 --hyp hyp.scratch.1280.yaml --epochs 300
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 tune.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6.yaml --weights 'runs/train/yolor-p6/weights/last_298.pt' --sync-bn --device 0,1,2,3,4,5,6,7 --name yolor-p6-tune --hyp hyp.finetune.1280.yaml --epochs 450
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6.yaml --weights 'runs/train/yolor-p6-tune/weights/epoch_424.pt' --sync-bn --device 0,1,2,3,4,5,6,7 --name yolor-p6-fine --hyp hyp.finetune.1280.yaml --epochs 450
```

## Inference

```
python detect.py --source images/horses.jpg --weights yolor-p6.pt --conf 0.25 --img-size 1280 --device 0
```

## Citation

```
@article{wang2021you,
  title={You Only Learn One Representation: Unified Network for Multiple Tasks},
  author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2105.04206},
  year={2021}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

</details>


## Export to onnx
```commandline
python models/export.py --weights=/path/to/your/weights.pt --img-size=1280
```
