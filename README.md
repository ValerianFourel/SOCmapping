

to use:
````bash
condor_submit_bid 1000 -i -append request_memory=481920 -append request_cpus=100 -append request_disk=200G -append request_gpus=3 -append 'requirements = CUDADeviceName == "NVIDIA A100-SXM4-80GB"'
```
