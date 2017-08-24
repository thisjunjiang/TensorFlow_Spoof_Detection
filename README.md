# TensorFlow_Spoof_Detection
Build neural networks using TensorFlow and run it in a distributed manner.

All ps tasks and worker tasks run the same codes but we run different command.

For the ps task:

python spoof_nn_dist.py --job_name=ps --task_index=0

For worker tasks:

python spoof_nn_dist.py --job_name=worker --task_index=0

python spoof_nn_dist.py --job_name=worker --task_index=1

python spoof_nn_dist.py --job_name=worker --task_index=2
