# copied from generation_inference.py

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.conversation import get_conv_template
import numpy as np
import os
import PIL.Image

import time
import contexttimer
import argparse

# autoregressive sampling
from sampling.autoregressive_sampling import autoregressive_sampling
# speculative sampling
#from sampling.speculative_sampling import speculative_sampling

MODELZOO = {
    'Janus-Pro-1B': 'models/Janus-Pro-1B',
    'Janus-Pro-7B': 'models/Janus-Pro-7B',
}

def parse_arguments():
    parser = argparse.ArgumentParser()

    # args from llm sp
    parser.add_argument('--input', type=str, default="A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.")
    parser.add_argument('--approx_model_name', type=str, default="Janus-Pro-1B")
    parser.add_argument('--target_model_name', type=str, default="Janus-Pro-7B")
    parser.add_argument('--approx_tmp', type=float, default=1.0, help='temperature for approx model')
    parser.add_argument('--target_tmp', type=float, default=1.0, help='temperature for target model')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    parser.add_argument('--test_time', '-t', type=int, default=10, help='number of measurements of the benchmark')

    # args from llamagen
    #parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    #parser.add_argument("--gpt-ckpt", type=str, default=None)
    #parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    #parser.add_argument("--from-fsdp", action='store_true')
    #parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    #parser.add_argument("--compile", action='store_true', default=False)
    #parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    #parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    #parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    #parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--max_new_tokens", type=int, default=576)
    parser.add_argument("--img-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--patch-size", type=int, choices=[8, 16], default=16)
    #parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    
    args = parser.parse_args()
    return args

# benchmark 
def benchmark(fn, print_prefix, test_time, use_profiler=True, *args, **kwargs):
    '''
    repeat the fn for [TEST_TIME] and test the average time cost
    '''
    TEST_TIME = test_time
    assert TEST_TIME > 0

    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / (t.elapsed / TEST_TIME)}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")


def prompt_to_tokens(chat_processor: VLChatProcessor, prompt: str):

    conv = get_conv_template('deepseek')
    conv.append_message(conv.roles[0], prompt.strip())
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + chat_processor.image_start_tag

    input_ids = chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    # parallel_size = 1
    tokens = torch.zeros((2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = chat_processor.pad_id
    print(tokens)
    return tokens

@torch.inference_mode()
def main(args):
    # load tokenizer
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    # text tokenizer of Janus-1B and Janus-7B is slightly different, but their image token codebooks might be the same
    approx_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.approx_model_name)
    target_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.target_model_name)

    approx_input_ids = prompt_to_tokens(approx_chat_processor, args.input)
    target_input_ids = prompt_to_tokens(target_chat_processor, args.input)
    
    # load gpt models
    print(f"begin loading models: \n {args.approx_model_name} \n {args.target_model_name}")
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    approx_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.approx_model_name, device_map = 'cuda', torch_dtype = precision, trust_remote_code=True
    )
    approx_model = approx_model.eval()

    target_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.target_model_name, device_map = 'cuda', torch_dtype = precision, trust_remote_code=True
    )
    target_model = target_model.eval()

    # autoregressive sampling
    torch.cuda.synchronize()
    t1 = time.time()
    index_sample = autoregressive_sampling(
        x=approx_input_ids, 
        model=approx_model, 
        N=args.max_new_tokens, 
        temperature=args.approx_tmp, 
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_weight=args.cfg_weight
        )
    torch.cuda.synchronize()
    sampling_time = time.time() - t1
    print(f"{args.approx_model_name} sampling takes about {sampling_time:.2f} seconds.")
    
    # decode
    dec = approx_model.gen_vision_model.decode_code(index_sample.to(dtype=torch.int), shape=[1, 8, args.img_size//args.patch_size, args.img_size//args.patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    t2 = time.time()
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # Save and display images:
    visual_img = np.zeros((1, args.img_size, args.img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    save_path = os.path.join('generated_samples', "approx_model_ar_sampling_img.jpg")
    PIL.Image.fromarray(visual_img[0]).save(save_path)
    print(f'image is saved to {save_path}')

    if args.benchmark:
        benchmark(autoregressive_sampling, 'approx model', test_time=args.test_time, use_profiler=False, x=approx_input_ids, model=approx_model, 
        N=args.max_new_tokens, temperature=args.approx_tmp, top_k=args.top_k,top_p=args.top_p,
        cfg_weight=args.cfg_weight)
    
    # autoregressive sampling
    torch.cuda.synchronize()
    t1 = time.time()
    index_sample = autoregressive_sampling(
        x=target_input_ids, 
        model=target_model, 
        N=args.max_new_tokens, 
        temperature=args.target_tmp, 
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_weight=args.cfg_weight
        )
    torch.cuda.synchronize()
    sampling_time = time.time() - t1
    print(f"{args.target_model_name} sampling takes about {sampling_time:.2f} seconds.")
    
    # decode
    dec = target_model.gen_vision_model.decode_code(index_sample.to(dtype=torch.int), shape=[1, 8, args.img_size//args.patch_size, args.img_size//args.patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    t2 = time.time()
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # Save and display images:
    visual_img = np.zeros((1, args.img_size, args.img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    save_path = os.path.join('generated_samples', "target_model_ar_sampling_img.jpg")
    PIL.Image.fromarray(visual_img[0]).save(save_path)
    print(f'image is saved to {save_path}')

    if args.benchmark:
        benchmark(autoregressive_sampling, 'target model', test_time=args.test_time, use_profiler=False, x=target_input_ids, model=target_model, 
        N=args.max_new_tokens, temperature=args.target_tmp, top_k=args.top_k,top_p=args.top_p,
        cfg_weight=args.cfg_weight)

    # speculative sampling
    # TODO

if __name__ == "__main__":
    args = parse_arguments()

    main(args)
