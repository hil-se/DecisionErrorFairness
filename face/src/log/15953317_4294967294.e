2023-04-28 11:04:17.975859: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38283 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2023-04-28 11:04:18.007265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 38283 MB memory:  -> device: 1, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:5e:00.0, compute capability: 8.0
2023-04-28 11:04:18.009133: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 38283 MB memory:  -> device: 2, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:86:00.0, compute capability: 8.0
2023-04-28 11:04:18.011294: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 38283 MB memory:  -> device: 3, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:af:00.0, compute capability: 8.0
Traceback (most recent call last):
  File "/home/zxyvse/DecisionErrorFairness/face/src/main.py", line 17, in <module>
    run()
  File "/home/zxyvse/DecisionErrorFairness/face/src/main.py", line 12, in run
    results = exp.run()
  File "/home/zxyvse/DecisionErrorFairness/face/src/rft.py", line 28, in run
    predicts = self.learn(X_train, y_train, X_test)
  File "/home/zxyvse/DecisionErrorFairness/face/src/rft.py", line 77, in learn
    model = VGG_Pre()
  File "/home/zxyvse/DecisionErrorFairness/face/src/vgg_pre.py", line 64, in __init__
    base_model.load_weights('checkpoint/vgg_face_weights.h5')
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-keras-2.10.0-mlh2nyw46ahbv4oxyqaqqnsf6tit6i7b/lib/python3.10/site-packages/keras/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-h5py-3.7.0-v4umzmspquyrhdo5gcbrcagewebgwp4d/lib/python3.10/site-packages/h5py/_hl/files.py", line 533, in __init__
    fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-skylake_avx512/gcc-11.2.0/py-h5py-3.7.0-v4umzmspquyrhdo5gcbrcagewebgwp4d/lib/python3.10/site-packages/h5py/_hl/files.py", line 226, in make_fid
    fid = h5f.open(name, flags, fapl=fapl)
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5f.pyx", line 106, in h5py.h5f.open
FileNotFoundError: [Errno 2] Unable to open file (unable to open file: name = 'checkpoint/vgg_face_weights.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
