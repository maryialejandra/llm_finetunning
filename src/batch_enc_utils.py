import torch as pt

from torch import Tensor

from transformers.tokenization_utils_base import BatchEncoding
from typing import Callable

def batch_enc_apply(batch_enc: BatchEncoding, fun: Callable[[Tensor], Tensor]) -> BatchEncoding:
    """Applies a function to each value in the batch encoding. And return a new BatchEncoding object with same keys."""
    return BatchEncoding({k: fun(v) for k, v in batch_enc.items()})


def batch_enc_cat(batch_enc1: BatchEncoding, batch_enc2: BatchEncoding, dim: int = 1) -> BatchEncoding:
    """Given two batch encodings with (hopefully the same keys)"""
    return BatchEncoding({k: pt.cat([batch_enc1[k], batch_enc2[k]], dim=dim)
                          for k in batch_enc1.keys()})

def pad_batch_1d_to_len(batch_enc: BatchEncoding, target_len: int,
                        pad_value: int | float,
                        attention_pad_value: int | float) -> BatchEncoding:
    """Aplica la función pad_1d_tensor_to_len para aumentar el tamaños de los tensores a asociados a las llaves
    `input_ids` y `attention_mask` de un batch encoding dado, hasta una longitud especificada por `target_len`.
    El valor pad_value es usado para `input_ids` y """
    return BatchEncoding({
        "input_ids": pad_1d_tensor_to_len(batch_enc['input_ids'],
                                          target_len, pad_value),
        "attention_mask": pad_1d_tensor_to_len(batch_enc['attention_mask'],
                                               target_len, attention_pad_value)
    })


def pad_1d_tensor_to_len(x: pt.Tensor, target_len: int, pad_value: int | float)-> pt.Tensor:
    """Aumentar el tamaño de un tensor 1-dimensional hasta la longitud  target_len, rellenando el resto de entradas
     `pad_value`. Si el tamaño de el vector ya es mayor que target_len se arrojará una excepción."""
    result = pt.ones(target_len, dtype=x.dtype) * pad_value
    result[:x.shape[0]] = x

    return result

def batch_shapes(batch_enc: BatchEncoding) -> dict[str, pt.Size]:
    """Returns the shape of each tensor in the batch encoding."""
    return {k: (
                v.shape if isinstance(v, pt.Tensor)
                else (
                    batch_shapes(v) if isinstance(v, BatchEncoding)
                    else None
                )
            ) for k, v in batch_enc.items()
            }