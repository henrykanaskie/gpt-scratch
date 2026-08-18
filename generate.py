import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = []
        for i in range(new_chars):
            # crop to context length
            if context.shape[1] > context_length:
                context = context[:, -context_length:]
            logits = model(context)
            last_char = logits[:, -1, :]
            pred = nn.functional.softmax(last_char, dim=-1)
            next_token = torch.multinomial(pred, 1, generator=generator)
            generator.set_state(initial_state)
            
            context = torch.cat((context, next_token), dim=-1)
            result.append(int_to_char[next_token.item()])
        return ''.join(result)


