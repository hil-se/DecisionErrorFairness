2023-08-20 09:33:31.904509: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38283 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2023-08-20 09:33:31.911645: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 38283 MB memory:  -> device: 1, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:5e:00.0, compute capability: 8.0
2023-08-20 09:33:31.913669: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 38283 MB memory:  -> device: 2, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:86:00.0, compute capability: 8.0
2023-08-20 09:33:31.916048: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 38283 MB memory:  -> device: 3, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:af:00.0, compute capability: 8.0
2023-08-20 09:33:40.527802: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8400
2023-08-20 09:33:44.261117: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
Traceback (most recent call last):
  File "/home/zxyvse/DecisionErrorFairness/face/src/main.py", line 13, in <module>
    run()
  File "/home/zxyvse/DecisionErrorFairness/face/src/main.py", line 7, in run
    exp.run()
  File "/home/zxyvse/DecisionErrorFairness/face/src/rft.py", line 84, in run
    result[A + ": " + "RBD"] = t_color + "%.2f" % m.RBD(self.data[A][train], self.data[A][test])
  File "/home/zxyvse/DecisionErrorFairness/face/src/biased_bridge.py", line 78, in RBD
    mu0, var0 = self.stats(group0_train, group0_test)
AttributeError: 'BiasedBridge' object has no attribute 'stats'
