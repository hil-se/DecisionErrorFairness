2023-08-19 11:12:52.572102: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38283 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2023-08-19 11:12:52.579856: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 38283 MB memory:  -> device: 1, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:5e:00.0, compute capability: 8.0
2023-08-19 11:12:52.581954: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 38283 MB memory:  -> device: 2, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:86:00.0, compute capability: 8.0
2023-08-19 11:12:52.584527: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 38283 MB memory:  -> device: 3, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:af:00.0, compute capability: 8.0
2023-08-19 11:12:59.865393: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8400
2023-08-19 11:13:03.462074: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
Traceback (most recent call last):
  File "/home/zxyvse/DecisionErrorFairness/face/src/main.py", line 13, in <module>
    run()
  File "/home/zxyvse/DecisionErrorFairness/face/src/main.py", line 7, in run
    exp.run()
  File "/home/zxyvse/DecisionErrorFairness/face/src/rft.py", line 83, in run
    t_color = self.t_str(m.RBT(self.data[A][train], self.data[A][test]))
  File "/home/zxyvse/DecisionErrorFairness/face/src/biased_bridge.py", line 28, in RBT
    mu0, var0 = self.distr_minus(self.delta_train[group0_train], self.delta_test[group0_test])
  File "/home/zxyvse/DecisionErrorFairness/face/src/biased_bridge.py", line 16, in distr_minus
    mu_train, var_train = self.norm_stats(num_train, ddof=ddof)
  File "/home/zxyvse/DecisionErrorFairness/face/src/biased_bridge.py", line 12, in norm_stats
    var = np.var(x, ddof)
  File "<__array_function__ internals>", line 5, in var
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-numpy-1.21.6-3cfi2mrirfn2syco34dzhvwwjttijeaw/lib/python3.10/site-packages/numpy/core/fromnumeric.py", line 3723, in var
    return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-numpy-1.21.6-3cfi2mrirfn2syco34dzhvwwjttijeaw/lib/python3.10/site-packages/numpy/core/_methods.py", line 199, in _var
    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-numpy-1.21.6-3cfi2mrirfn2syco34dzhvwwjttijeaw/lib/python3.10/site-packages/numpy/core/_methods.py", line 76, in _count_reduce_items
    items *= arr.shape[mu.normalize_axis_index(ax, arr.ndim)]
numpy.AxisError: axis 1 is out of bounds for array of dimension 1
