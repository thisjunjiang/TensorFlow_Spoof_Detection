# TensorFlow_Spoof_Detection
Build neural networks using TensorFlow and run it in a distributed manner.

All ps tasks and worker tasks run the same codes but we run different command.

For the ps task:

python spoof_nn_dist.py --job_name=ps --task_index=0

For worker tasks:

python spoof_nn_dist.py --job_name=worker --task_index=0

python spoof_nn_dist.py --job_name=worker --task_index=1

python spoof_nn_dist.py --job_name=worker --task_index=2


Some plots from experiments

![two_node_tf](https://user-images.githubusercontent.com/14324327/29692553-935bd0c0-88e5-11e7-919b-32dccda548a3.png)
![three_node_tf](https://user-images.githubusercontent.com/14324327/29692668-119d6688-88e6-11e7-9b78-2b483fc82659.png)
![two_three_node_tf](https://user-images.githubusercontent.com/14324327/29692670-14bdd35c-88e6-11e7-8b27-3775239b885c.png)
