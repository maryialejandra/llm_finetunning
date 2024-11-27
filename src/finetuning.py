import math
import torch as pt
from torch import nn


def freeze_and_install_lora(model: nn.Module, *, lora_rank: int):
    # Congelamos todos los parámetros del modelo
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazamos las capas lineales del mecanismo de atención por capas LoRA
    for layer in model.model.layers:
        if hasattr(layer, 'self_attn'):
            layer.self_attn.q_proj = LoraLinear(layer.self_attn.q_proj, r=lora_rank)
            layer.self_attn.k_proj = LoraLinear(layer.self_attn.k_proj, r=lora_rank)
            layer.self_attn.v_proj = LoraLinear(layer.self_attn.v_proj, r=lora_rank)
            layer.self_attn.o_proj = LoraLinear(layer.self_attn.o_proj, r=lora_rank)

    show_lora_param_stats(model)


class LoraLinear(nn.Module):
    def __init__(self, linear_layer: nn.Module, r: int = 1):
        super().__init__()
        # Se cambia el tipo de la capa a float32 para evitar errores durante el entrenamiento:
        self.linear_layer = linear_layer.to(pt.float32)
        # Se guarda el valor de r (rank)
        self.r = r
        # Se calculan los valores de fan_in y fan_out para inicializar los valores de A y B
        fan_in = self.linear_layer.in_features
        fan_out = self.linear_layer.out_features
        # Se crean las matrices A y B de la técnica de Low Rank Adaptation
        self.lora_A = nn.Parameter(pt.zeros((fan_in, r)))
        self.lora_B = nn.Parameter(pt.zeros((r, fan_out)))
        # Se inicializa la matriz A con una distribución uniforme
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Se congelan los pesos originales de la capa lineal
        self.linear_layer.weight.requires_grad = False

    # Este método se llama cuando se cambia el modo de entrenamiento
    # del modelo (model.train() o model.eval())
    def train(self, mode=True):
        self.training = mode
        # En inferencia se fusionan los pesos de la capa lineal y los LoRA
        if not mode:
            self.merged_weight = (
                self.linear_layer.weight.transpose(0,1)
                + self.lora_A @ self.lora_B
            ).to(pt.float16) # Se cambia el tipo de la matriz a float16

    def forward(self, x):
        # En entrenamiento se suman los resultados de la capa lineal y los de LoRA
        if self.training:
            x = x.to(pt.float32)
            # Se calcula la salida de la capa lineal
            output = self.linear_layer(x)
            # Se calcula la salida de LoRA y se suman
            output += x @ self.lora_A @ self.lora_B
            output = output.to(pt.float16)
        else:
            output = x @ self.merged_weight
        return output


def show_lora_param_stats(model: nn.Module):
    n_params_without_lora = 0
    n_params_with_lora = 0
    for name, param in model.named_parameters():
        if 'self_attn' in name and 'linear_layer' in name:
            n_params_without_lora += param.numel()
        if param.requires_grad:
            n_params_with_lora += param.numel()

    pct_with_lora = 100 * n_params_with_lora / (n_params_without_lora + n_params_with_lora)
    print(f'Parámetros sin LoRA: {n_params_without_lora:,} || Parámetros con LoRA: {n_params_with_lora:,} '
          f' || Porcentaje de parámetros con LoRA: {pct_with_lora:.2f}%')
