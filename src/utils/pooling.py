import torch


class LastTokenPooling:
    def __init__(self, layer_number: int = -1):
        self.layer_number = layer_number

    def __call__(self, hidden_states, input_ids, model):
        # hs = hidden_states[self.layer_number]
        hs = hidden_states
        bs = input_ids.shape[0]

        non_pad_mask = (input_ids != model.config.pad_token_id).to(device=hs.device, dtype=torch.int32)
        token_indices = torch.arange(input_ids.shape[-1], device=hs.device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        pooled_logits = hs[torch.arange(bs, device=hs.device), last_non_pad_token]
        return pooled_logits


class CLSTokenPooling:
    def __init__(self, layer_number: int = -1):
        self.layer_number = layer_number

    def __call__(self, hidden_states, input_ids, model):
        hs = hidden_states[self.layer_number]
        return hs[:, 0, :]
