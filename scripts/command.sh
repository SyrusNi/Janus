# slurm
srun --pty -N 1 -n 1 -p a800 -q normal --gres=gpu:1 -t 4:00:00 /bin/bash

# generation
python generation_inference.py

python -m ipdb main.py --approx_model_name models/Janus-Pro-1B --target_model_name models/Janus-Pro-7B
python main.py --approx_model_name models/Janus-Pro-1B --target_model_name models/Janus-Pro-7B -b -t 3
python main.py --approx_model_name models/Janus-Pro-1B --target_model_name models/Janus-Pro-7B --approx_tmp 0.0 --target_tmp 1.0
python main.py --approx_model_name models/Janus-Pro-1B --target_model_name models/Janus-Pro-7B --approx_tmp 0.0 --target_tmp 0.0
python main.py --approx_model_name models/Janus-Pro-1B --target_model_name models/Janus-Pro-7B --precision fp16
python main.py --approx_model_name models/Janus-Pro-1B --target_model_name models/Janus-Pro-7B --precision none --gamma 1

python -m ipdb main.py --approx_model_name models/Janus-Pro-1B --target_model_name models/Janus-Pro-1B