2023-05-08 16:23:38.383121: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38283 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2023-05-08 16:23:38.386864: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 38283 MB memory:  -> device: 1, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:5e:00.0, compute capability: 8.0
2023-05-08 16:23:38.388921: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 38283 MB memory:  -> device: 2, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:86:00.0, compute capability: 8.0
2023-05-08 16:23:38.391059: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 38283 MB memory:  -> device: 3, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:af:00.0, compute capability: 8.0
2023-05-08 16:23:44.316277: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8400
2023-05-08 16:23:46.575730: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
Traceback (most recent call last):
  File "/home/zxyvse/DecisionErrorFairness/face/src/main.py", line 14, in <module>
    run()
  File "/home/zxyvse/DecisionErrorFairness/face/src/main.py", line 10, in run
    exp.run()
  File "/home/zxyvse/DecisionErrorFairness/face/src/rft.py", line 33, in run
    for i in range(m):
TypeError: 'Metrics' object cannot be interpreted as an integer
