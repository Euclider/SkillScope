import torch

from phase2.measure import counter_input


class MinimalTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert tokenize is False  # Explicitly avoid Transformers 5's dict return default.
        return "U:"+messages[0]["content"]+":A"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("ascii"))


def test_counterprompt_keeps_old_response_and_handles_null_padding():
    tokenizer = MinimalTokenizer()
    prompt = "state old payload actions"
    ids = tokenizer.encode("U:"+prompt+":A", add_special_tokens=False)
    width = len(ids)+10
    original = torch.tensor([0]*10+ids+[7,8])
    mask = torch.tensor([0]*10+[1]*len(ids)+[1,1])
    tensors = {"responses":torch.tensor([[7,8]]), "input_ids":original[None],
               "attention_mask":mask[None]}
    metadata = {"decision_id":"d", "info":{"prompt_text":prompt,"phase2_payload_text":"old payload"}}
    for arm, replacement in (("placebo","new content"),("null","")):
        changed, changed_mask, position = counter_input(tokenizer,metadata,tensors,0,arm,"new content")
        assert changed[-2:].tolist() == [7,8]
        assert changed_mask[-2:].tolist() == [1,1]
        expected = tokenizer.encode("U:"+prompt.replace("old payload",replacement)+":A",add_special_tokens=False)
        assert changed[:width][changed_mask[:width].bool()].tolist() == expected
        assert position[-1].item() == len(expected)+1
    torch.testing.assert_close(tensors["input_ids"][0],original)
