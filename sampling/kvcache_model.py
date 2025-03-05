import torch

from sampling.utils import norm_logits, sample
from janus.models.modeling_vlm import MultiModalityCausalLM
from transformers.cache_utils import DynamicCache


class KVCacheModel():
    def __init__(self, model : MultiModalityCausalLM, temperature : float = 1, top_k : int = 0, top_p : float = 0, cfg_weight : float = 5.0, **kwargs) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._cfg_weight = cfg_weight


    def prefill(self, input_ids : torch.Tensor):
        # deal with text tokens
        assert self._past_key_values is None
        inputs_embeds = self._model.language_model.get_input_embeddings()(input_ids)
        outputs = self._model.language_model.model(inputs_embeds = inputs_embeds, use_cache = True)
        logits = self.lm_head(outputs.last_hidden_state)
        self._prob_history = norm_logits(logits, temperature=self._temperature, top_p=self._top_p, top_k=self._top_k)
        self._past_key_values = outputs.past_key_values
        last_q = self._prob_history[:, -1, :]
        next_token = sample(last_q)
        next_token = torch.cat([next_token, next_token], dim=0)
        output_ids = torch.cat([input_ids, next_token], dim=1)

        return output_ids


    def lm_head(self, hidden_states: torch.Tensor):
        b, s, h = hidden_states.size()
        logits = self._model.gen_head(hidden_states.reshape(b*s, h))
        logits = logits.reshape(b, s, -1)
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + self._cfg_weight * (logit_cond-logit_uncond) # [b, v]

        return logits


    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        assert self._prob_history is not None
        cached_len = self._past_key_values.get_seq_length()
        
        if cached_len == input_ids.shape[1]:
            last_q = self._prob_history[:, -1, :]
            return last_q
        
        last_input_id = input_ids[:, cached_len:]
        
        inputs_embeds = self._model.prepare_gen_img_embeds(last_input_id)
        outputs = self._model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self._past_key_values)
        logits = self.lm_head(outputs.last_hidden_state) # [b, s, v]
        
        probs = norm_logits(logits, temperature=self._temperature, top_p=self._top_p, top_k=self._top_k) # [b, s, v]
        self._prob_history = torch.cat([self._prob_history, probs], dim=1)
        self._past_key_values = outputs.past_key_values
        last_q = self._prob_history[:, -1, :]
        
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = sample(q)
            next_tok = torch.cat([next_tok, next_tok], dim=0)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        k_cache = []
        v_cache = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv

            # k, v (batch, head, seq, hidden_dim)
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            k_cache.append(k)
            v_cache.append(v)
        
        self._past_key_values = DynamicCache()
        self._past_key_values.key_cache = k_cache
        self._past_key_values.value_cache = v_cache
        self._prob_history = self._prob_history[:, :end_pos, :]

