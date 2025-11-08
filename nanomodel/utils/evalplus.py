import types

from transformers import AutoTokenizer, PreTrainedModel


def patch_strip(self, *args, **kwargs):
    return self.config.name_or_path.strip(*args, **kwargs)

def patch_tostring(self):
    return self.config.name_or_path

def patch_evalplus(model):
    from ..models.base import BaseNanoModel
    if isinstance(model, BaseNanoModel) or isinstance(model, PreTrainedModel):
        model.strip = types.MethodType(patch_strip, model)
        model.__str__ = types.MethodType(patch_tostring, model)
        model.__repr__ = types.MethodType(patch_tostring, model)

    import torch
    from evalplus.provider.base import DecoderBase
    # from evalplus.provider.nanomdel import NanoModelDecoder
    from evaplus_nanomodel import NanoModelDecoder
    from evalplus.provider.utility import extra_eos_for_direct_completion

    from .. import AutoNanoModel
    from ..models import BaseNanoModel

    class PatchedNanoModelDecoder(DecoderBase):
        def __init__(
                self,
                name: str,
                dataset: str,
                nanomodel_backend: str = 'auto',
                force_base_prompt: bool = False,
                **kwargs,
        ):

            super(NanoModelDecoder, self).__init__(name=name, **kwargs)

            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                device = torch.device("mps")
            elif hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
                device = torch.device("xpu")
            elif hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            self.device = device

            kwargs = {
                "model_id_or_path": name,
                "trust_remote_code": self.trust_remote_code,
                "backend": nanomodel_backend,
                "device": device
            }
            self.skip_special_tokens = True
            self.force_base_prompt = force_base_prompt
            if isinstance(name, BaseNanoModel):
                self.model = name
                self.tokenizer = self.model.tokenizer
            elif isinstance(name, PreTrainedModel):
                self.model = name
                self.tokenizer = AutoTokenizer.from_pretrained(
                    name.config.name_or_path,
                    trust_remote_code=self.trust_remote_code,
                )
            elif isinstance(name, str):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    name,
                    trust_remote_code=self.trust_remote_code,
                )
                self.model = AutoNanoModel.load(**kwargs)
                self.model = self.model.to(self.device)
            else:
                raise ValueError(f"`name` is invalid. expected: `model instance or str` actual: `{name}`")

            if self.tokenizer is None:
                raise ValueError("Tokenizer: Auto-loading of tokenizer failed with `model_or_id_or_path`. Please pass in `tokenizer` as argument.")

            if self.is_direct_completion():  # no chat template
                self.eos += extra_eos_for_direct_completion(dataset)
            else:  # with chat template
                self.eos += ["\n```\n"]

        def __str__(self):
            if isinstance(self.model, str):
                return self.model
            elif isinstance(self.model, PreTrainedModel):
                return self.model.config.name_or_path
            elif isinstance(self.model, BaseNanoModel):
                return self.model.model_local_path
            else:
                return self.model.__class__.__name__


    NanoModelDecoder.__init__ = PatchedNanoModelDecoder.__init__
    NanoModelDecoder.__str__ = PatchedNanoModelDecoder.__str__
