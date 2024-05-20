from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers

from lm_eval.base import BaseLM


class HFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        torch_dtype=None,
        device_map=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        use_fast: Optional[bool] = True,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
            device_map=device_map,
            revision=revision,
            trust_remote_code=trust_remote_code,
        ).eval()
        if not load_in_8bit:
            try:
                self.gpt2.to(self.device)
            except:  # noqa: E722
                print(
                    "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore."
                )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.gpt2.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            do_sample=False,
        )


class HFFLAXLM(BaseLM):
    def __init__(
        self,
        pretrained,
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        trust_remote_code: Optional[bool] = False,
        use_fast: Optional[bool] = False,
    ):
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.model = transformers.FlaxAutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
        self.vocab_size = self.tokenizer.vocab_size
        self.batch_size_per_gpu = batch_size
        self.count = 0

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_len

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return torch.device("cpu")

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def jax_to_torch(self, jax_array):
        numpy_array = np.array(jax_array, dtype=np.float32)
        torch_tensor = torch.from_numpy(numpy_array)
        return torch_tensor

    def torch_to_jax(self, torch_tensor):
        numpy_array = torch_tensor.numpy()
        jax_array = jnp.array(numpy_array)
        return jax_array

    def _model_call(self, inps):
        jax_inps = self.torch_to_jax(inps)

        model_kwargs = self.model.prepare_inputs_for_generation(jax_inps, self.max_length, None)
        model_outputs = self.model(jax_inps, params=None, **model_kwargs)

        # Workaround: avoid out of memory error
        self.count = self.count + 1
        if self.count % 100 == 0:
            jax.clear_caches()

        logits = model_outputs["logits"]
        torch_logits = self.jax_to_torch(logits)
        return torch_logits

    def _model_generate(self, context, max_length, eos_token_id):
        input_ids = self.torch_to_jax(context)

        output = self.model.generate(
            input_ids,
            max_length=max_length,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            do_sample=False,
        )

        # Workaround: avoid out of memory error
        self.count = self.count + 1
        if self.count % 100 == 0:
            jax.clear_caches()

        return output[0]


# for backwards compatibility
GPT2LM = HFLM
