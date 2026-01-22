# learn-cuda

Learning CUDA, following different resources.

## An Even Easier Introduction to CUDA (Updated)

Notes from following https://developer.nvidia.com/blog/even-easier-introduction-cuda/.

To compile the CUDA program run
```
nvcc add.cu -o add_cuda
```
You can run that program with
```
./add_cuda t 30
```
It's interesting to watch how the memory moves from the Linux process memory to the GPU.
At first the memory counts towards RssFile of the process:
```
awitt@awitt-pc:~/code/learn-cuda$ ./add_cuda t 29 &
[1] 23396
awitt@awitt-pc:~/code/learn-cuda$ cat /proc/23396/status | grep -i rss
VmRSS:	 4302192 kB
RssAnon:	   22488 kB
RssFile:	 4271512 kB
RssShmem:	    8192 kB
```

Then it starts counting towards GPU memory (although confusingly, not to the add_cuda process):
```
$ nvidia-smi
Thu Jan 22 16:26:47 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     On  |   00000000:01:00.0 Off |                  N/A |
|  0%   34C    P8              4W /  180W |    8343MiB /  16311MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3044      G   /usr/lib/xorg/Xorg                        4MiB |
|    0   N/A  N/A           22818      C   ./add_cuda                              128MiB |
+-----------------------------------------------------------------------------------------+

```
Then it goes back to counting towards the process memory:
```
$ cat /proc/23396/status | grep -i rss
VmRSS:	 2205532 kB
RssAnon:	   22532 kB
RssFile:	 2174808 kB
RssShmem:	    8192 kB
```
But only the memory that's needed.  This is using Unified Memory.


To profile it, run:
```
nsys profile -t cuda  -o report --force-overwrite=true --stats=true ./add_cuda f 20
```
That prints info about the program and also writes a SQLite file.  To analyze that data, you can use SQLite or DuckDB.  For example:
```
D install sqlite; load sqlite;
D attach 'report1.sqlite' as report (type sqlite);
D use report;
```
The tables in that database store profile info.
