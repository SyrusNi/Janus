# slurm
srun --pty -N 1 -n 1 -p a800 -q normal --gres=gpu:1 -t 4:00:00 /bin/bash

# generation
python generation_inference.py

python -m ipdb main.py --approx_model_name models/Janus-Pro-1B --target_model_name models/Janus-Pro-7B
python main.py --approx_model_name models/Janus-Pro-1B --target_model_name models/Janus-Pro-7B
python main.py --approx_model_name models/Janus-Pro-1B --target_model_name models/Janus-Pro-7B --precision fp16