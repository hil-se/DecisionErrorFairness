2023-05-02 08:33:36.040487: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38283 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2023-05-02 08:33:36.050028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 38283 MB memory:  -> device: 1, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:5e:00.0, compute capability: 8.0
2023-05-02 08:33:36.052162: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 38283 MB memory:  -> device: 2, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:86:00.0, compute capability: 8.0
2023-05-02 08:33:36.054534: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 38283 MB memory:  -> device: 3, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:af:00.0, compute capability: 8.0
2023-05-02 08:33:44.189212: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8400
2023-05-02 08:33:47.208668: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
Traceback (most recent call last):
  File "/home/zxyvse/DecisionErrorFairness/face/src/main.py", line 13, in <module>
    run("P1")
  File "/home/zxyvse/DecisionErrorFairness/face/src/main.py", line 8, in run
    results = exp.run(base)
  File "/home/zxyvse/DecisionErrorFairness/face/src/rft.py", line 52, in run
    result[A + ": " + "CBD"] = "%.2f" %m.CBD(self.data[A][test])
  File "/home/zxyvse/DecisionErrorFairness/face/src/metrics.py", line 69, in CBD
    sigma = np.std(self.y_pred - self.y)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-pandas-1.5.3-b37yx5pszgtk777ocqv6ep5n54izgozn/lib/python3.10/site-packages/pandas/core/generic.py", line 2113, in __array_ufunc__
    return arraylike.array_ufunc(self, ufunc, method, *inputs, **kwargs)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-pandas-1.5.3-b37yx5pszgtk777ocqv6ep5n54izgozn/lib/python3.10/site-packages/pandas/core/arraylike.py", line 265, in array_ufunc
    result = maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
  File "pandas/_libs/ops_dispatch.pyx", line 113, in pandas._libs.ops_dispatch.maybe_dispatch_ufunc_to_dunder_op
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-pandas-1.5.3-b37yx5pszgtk777ocqv6ep5n54izgozn/lib/python3.10/site-packages/pandas/core/ops/common.py", line 72, in new_method
    return method(self, other)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-pandas-1.5.3-b37yx5pszgtk777ocqv6ep5n54izgozn/lib/python3.10/site-packages/pandas/core/arraylike.py", line 114, in __rsub__
    return self._arith_method(other, roperator.rsub)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-pandas-1.5.3-b37yx5pszgtk777ocqv6ep5n54izgozn/lib/python3.10/site-packages/pandas/core/series.py", line 6259, in _arith_method
    return base.IndexOpsMixin._arith_method(self, other, op)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-pandas-1.5.3-b37yx5pszgtk777ocqv6ep5n54izgozn/lib/python3.10/site-packages/pandas/core/base.py", line 1327, in _arith_method
    return self._construct_result(result, name=res_name)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-pandas-1.5.3-b37yx5pszgtk777ocqv6ep5n54izgozn/lib/python3.10/site-packages/pandas/core/series.py", line 3223, in _construct_result
    out = self._constructor(result, index=self.index)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-pandas-1.5.3-b37yx5pszgtk777ocqv6ep5n54izgozn/lib/python3.10/site-packages/pandas/core/series.py", line 470, in __init__
    data = sanitize_array(data, index, dtype, copy)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-pandas-1.5.3-b37yx5pszgtk777ocqv6ep5n54izgozn/lib/python3.10/site-packages/pandas/core/construction.py", line 647, in sanitize_array
    subarr = _sanitize_ndim(subarr, data, dtype, index, allow_2d=allow_2d)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-pandas-1.5.3-b37yx5pszgtk777ocqv6ep5n54izgozn/lib/python3.10/site-packages/pandas/core/construction.py", line 698, in _sanitize_ndim
    raise ValueError("Data must be 1-dimensional")
ValueError: Data must be 1-dimensional
