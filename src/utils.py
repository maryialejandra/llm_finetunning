import gc
import math
import os
import random
import re
import time

from pathlib import Path
from typing import Generator, TypeVar, Callable


import huggingface_hub as hf_hub
import numpy as np
import torch as pt
from torch import nn


T = TypeVar('T')


LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}

def letter_to_idx(let: str) -> int:
    return LETTER_TO_IDX[let]


def find_first_idx(a_list: list[T], boolean_fun: Callable[[T], bool]) -> int | None:
    """Find index of first element of a_list for which boolean_fun(a_list[idx]) is True.
    Returns None if there is no such idx"""
    idx = None

    for i, a in enumerate(a_list):
        if boolean_fun(a):
            idx = i
            break

    return idx

def find_last_idx(a_list: list[T], boolean_fun: Callable[[T], bool]) -> int | None:
    """Find index of last element of a_list for which boolean_fun(a_list[idx]) is True.
    Returns None if there is no such idx"""
    idx = None

    for i, a in enumerate(a_list):
        if boolean_fun(a):
            idx = i

    return idx


def random_permutation(a_set: list[str|int]):
    dic = {k: random.uniform(0, 1) for k in a_set}
    return sorted(dic, key=dic.get)

def chunk_generator(lines: list[str], start_idx: int,
                    min_chunk_len: int) -> Generator[tuple[int, int, str], None, None]:

    n_lines = len(lines)
    end_idx = start_idx + 1

    def len_chunk():
        return sum((len(line) for line in lines[start_idx:end_idx]))

    while end_idx < n_lines:
        while start_idx >= end_idx or (len_chunk() < min_chunk_len and end_idx < n_lines):
            end_idx = min(end_idx + 1, n_lines)

        yield start_idx, end_idx, "\n".join(lines[start_idx:end_idx])

        start_idx += 1
        while start_idx < n_lines and lines[start_idx].strip() == "":
            # print("Advancing start_idx")
            start_idx += 1


def get_secret(var_name: str,  value: str = None,
               use_env: bool=True,
               use_colab: bool=True):
    if value is not None:
        print(f"Returning secret from passed argument: {value[:2]}...{value[-2:]}", )
        return value

    if use_env:
        value = os.environ.get(var_name)
        if value is not None:
            print(f"Returning secret from environment variable `{var_name}`=`{value[:2]}...{value[-2:]}`", )
            return value
        else:
            print(f"Secret {var_name} not found in env_var")

    if use_colab:
        from google.colab import userdata
        value = userdata.get(var_name)
        if value is not None:
            print(f"Returning google.colab secret `{var_name}`=`{value[:2]}...{value[-2:]}`", )
        else:
            print(f"Secret {var_name} not found in google.colab secrets")

    if value is None:
        raise ValueError(f"Value for secret named `{var_name}` not found! "
                         f"try switching one of the following flags to true use_env={use_env} use_colab={use_colab}")

def login_to_hf_hub(hf_token: str | None = None):
    """Will attempt to get HF-token first from google.colab.userdata, then from env var,
    and finally from passed argument.
    If successful login to huggingface hub with said token.
    """

    hf_token = get_secret("HF_TOKEN", hf_token)
    hf_hub.login(token=hf_token)

def module_device(mod: nn.Module) -> str:
    return next(mod.parameters()).device.type

def is_close(x: float, y: float, tol=1e-6) -> bool:
    return math.fabs(x - y) < tol

def free_gpu_memory() -> None:
    print(f"before freeing: {gpu_mem_info()}")
    gc.collect()
    pt.cuda.empty_cache()
    print(f"after  freeing: {gpu_mem_info()}")

def gpu_mem_info() -> dict[str, float]:
    free, total = pt.cuda.mem_get_info()
    return {
        "gpu_mem_used_MB": np.round((total - free) / 1e6, 2),
        "gpu_mem_free_MB": np.round(free / 1e6, 2),
        "gpu_mem_total_MB": np.round(total / 1e6, 2)
    }

def plot_loss_curves(train_losses: list[float],
                     valid_losses: list[float],
                     title: str,
                     steps_per_point: int | None = None):
    import matplotlib.pyplot as plt
    n_epochs = len(train_losses)

    if steps_per_point is not None:
        x_ticks = (pt.arange(n_epochs) + 1) * steps_per_point
    else:
        x_ticks = pt.arange(n_epochs) + 1
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    ax.plot(x_ticks,  train_losses, 'b.-')
    ax.plot(x_ticks, valid_losses, '.-', color='#ff9900')

    ax = plt.gca()
    ax.set_xticks(x_ticks)

    if steps_per_point is not None:
        ax.set_xlabel('Steps')
    else:
        ax.set_xlabel('Epoch #')
    ax.set_ylabel('Loss')
    plt.legend(['train set loss',
                'validation set loss'])

    plt.title(title)


def globals_report():
  globals_snapshot = globals().copy()
  for var_name, v in globals_snapshot.items():
    type_name = type(v).__name__

    if type_name in ('str', 'int', 'module', 'method', 'function', 'type'):
        continue

    print(f"{var_name:32s} : {type_name}")

    if re.match("^_[0-9]+$", var_name):
        print(f'DELETING {var_name} - type:{type_name}')
        del globals()[var_name]


def upload_file_to_s3(fpath: Path | str,
                      bucket: str = 'ml-school-teo',
                      s3_session = None) -> None:
    try:
        import boto3

        s3_session = s3_session or boto3.Session(
              aws_access_key_id=get_secret('AWS_KEY_ID'),
              aws_secret_access_key=get_secret('AWS_SECRET_KEY')
        )
        s3_client = s3_session.client('s3')

        fpath = Path(fpath)

        t0 = time.perf_counter()
        file_size_mb = fpath.lstat().st_size/1e6
        print(f"Uploading file: {fpath} size: {file_size_mb:.2f} MB to: s3://{bucket}/{fpath.name} ...")
        s3_client.upload_file(str(fpath), bucket, Key=fpath.name)
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f} secs - {file_size_mb / elapsed:.2f} MB/s")

    except Exception as exc:
        print(f"Error uploading file to S3: {exc}")

