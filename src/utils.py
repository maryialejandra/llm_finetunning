from typing import Generator, TypeVar, Callable

import random

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
            print("Advancing start_idx")
            start_idx += 1