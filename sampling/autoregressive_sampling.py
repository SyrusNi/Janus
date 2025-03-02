import torch

from sampling.utils import norm_logits, sample
from janus.models import MultiModalityCausalLM

@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : MultiModalityCausalLM, N : int, # prefix and model
                            temperature : float = 1, top_k : int = 0, top_p : float = 0, # sampling kwargs
                            cfg_weight : float = 5): # condition-free guidance
    '''autoregressive sampling
    Args:
        x: tokenized prompts. [b, s]
        model: ar model
        N: max_new_tokens
        sampling kwargs: temperature, top_k, top_p
        cfg kwargs: cfg_weight
    Return:
        generated tokens
    '''
    inputs_embeds = model.language_model.get_input_embeddings()(x)
    generated_tokens = torch.zeros((x.shape[0] // 2, N), dtype=torch.int).to(model.device)
    
    for i in range(N):
        outputs = model.language_model.model(inputs_embeds = inputs_embeds, use_cache = True, past_key_values = outputs.past_key_values if i != 0 else None)
        
        hidden_states = outputs.last_hidden_state

        logits = model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)

        probs = norm_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        next_token = sample(probs)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = model.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    return generated_tokens

