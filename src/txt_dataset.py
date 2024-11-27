import math
from pathlib import Path

import re

from torch import Tensor
from torch.utils.data import Dataset

import src.batch_enc_utils  as beut
from transformers import AutoTokenizer


def concat_files_to_str(plain_text_fpaths: list[Path]) -> str:

    all_lines = []
    for plain_text_fpath in plain_text_fpaths:
        with open(plain_text_fpath, 'r', encoding='utf-8') as f_in:
            file_lines: list[str] = f_in.readlines()
            file_size: int = plain_text_fpath.lstat().st_size  # in bytes
            print(f"{str(plain_text_fpath):96s}: {file_size:7,d} bytes, {len(file_lines):6,d} lines")
            all_lines.extend(prefix_line(line) for line in file_lines)

    ret = "\n".join(all_lines)

    debug_fpath = Path("tokenized_txt_dataset.concat.txt")
    with debug_fpath.open("wt") as f_out:
        f_out.write(ret)

    print(f"concat_files_to_str: returning {len(ret):6,d} characters, whole text at: {debug_fpath!s}")
    return ret


def prefix_line(line: str) -> str:
    if re.search(r"(Article|Art\.|Artículo) *([0-9]+)", line):
        return f"Reglamento de la Universidad de los Andes Regulations: {line}"
    else:
        return line

class TokenizedTxtDataset(Dataset):
    def __init__(self,
                 plain_text_fpaths: list[Path],
                 *,
                 block_size: int,
                 stride: int,
                 tokenizer: AutoTokenizer) -> None:

        all_text: str = concat_files_to_str(plain_text_fpaths)

        assert block_size % stride == 0, "ERROR: stride should divide block_size evenly...!"
        self.block_size: int = block_size
        self.max_stride_mult = block_size // stride

        print(f"TokenizedTxtDs.__init__: len(all_text)={len(all_text):,d}")
        token_batch = tokenizer(all_text, return_tensors='pt')

        raw_input_ids = token_batch['input_ids'][0]
        n_toks_raw: int = raw_input_ids.shape[0]

        self.n_blocks: int = int(math.ceil(n_toks_raw / block_size))
        n_toks_padded: int = block_size * self.n_blocks

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.input_ids = beut.pad_1d_tensor_to_len(raw_input_ids,
                                                   target_len=n_toks_padded,
                                                   pad_value=pad_token_id)
        self.attention_mask = beut.pad_1d_tensor_to_len(
                                        token_batch['attention_mask'][0],
                                        target_len=n_toks_padded,
                                        pad_value=0
                              )
        print(f"n_toks_raw: {n_toks_raw:7,d}  input_ids length (padded): {self.input_ids.shape[0]:7,d} "
              f"block_size: {self.block_size} n_blocks: {self.n_blocks}")

        self.total_toks = n_toks_padded
        self.stride: int = stride


    def __len__(self) -> int:
        return self.n_blocks * self.max_stride_mult

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        item, _ = self._get_item_and_offset(idx)
        return item

    def _get_item_and_offset(self, idx: int) -> tuple[dict[str, Tensor], int]:
        """Función que hace todo el trabajo y además retorna el offset.
           El offset lo usamos para testing"""
        block_idx = idx // self.max_stride_mult
        mult_stride = idx % self.max_stride_mult
        offset = block_idx * self.block_size + mult_stride * self.stride

        end_idx = offset + self.block_size

        # assert end_idx <= self.total_toks,\
        #     f"end_idx={end_idx} idx: {idx} block_idx={block_idx} mult_stride={mult_stride} offset={offset}"
        if end_idx > self.total_toks:
            offset = self.total_toks - self.block_size
            end_idx = self.total_toks

        return ({'input_ids': self.input_ids[offset:end_idx],
                 'attention_mask': self.attention_mask[offset:end_idx]},
                 offset)
