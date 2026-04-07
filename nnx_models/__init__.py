from .wire import WIRE, ComplexGaborLayer, RealGaborLayer
from .siren import SIREN, SirenLayer
from .mlp import MLP
from .mlp_plus import MLP_Plus
from .lora import LoRA, LoRALinear, add_lora_to_model, remove_lora_from_model, reset_lora_params, merge_lora_params, apply_lora_state

model_fn = {
    "mlp": MLP,
    "mlp+": MLP_Plus,
    "siren": SIREN,
    "wire": WIRE,
}