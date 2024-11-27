import re
from collections import Counter
from pathlib import Path

def process_raw_lines_estatutos(raw_lines: list[str]):
    """Preprocess raw text file:
      data/Estatutos-Universidad-de-los-Andes-2020-ratificados-MEN-RQ.raw.txt
      that was obtained via online conversion of PDF file of same name
    """
    total_chars_in = sum(len(line) for line in raw_lines)
    print(f"input has {len(raw_lines)} lines, {total_chars_in} chars")

    skip_counts = Counter()
    current_line = ""
    out_lines = []
    for i, line in enumerate(raw_lines):
        if line == "":
            out_lines.append(line)
            continue

        if line.startswith("\x0c"):
            # print(f"skipping special line {i}: `{line}`")
            skip_counts["special line \x0c"] += 1
        elif re.search("^[0-9]+\.", line):
            out_lines.append(current_line)
            current_line = line
        elif re.match("^Artículo +[0-9]+(.º)?$", line): # <h3>
            out_lines.append(current_line)
            out_lines.append(f"\n## {line}")
            current_line = ""
        elif re.match("^CAPÍTULO +[XVILCM]+$", line): # <h2>
            out_lines.append(current_line)
            out_lines.append(f"\n\n# {line}")
            current_line = ""

        elif line.strip()=="UNIVERSIDAD DE LOS ANDES":
            skip_counts["UNIVERSIDAD DE LOS ANDES"] += 1
            continue
        elif re.match("^[0-9]+$", line):
            skip_counts["single-number-line"] += 1
        elif line[0].lower() == line[0]:
            if current_line.endswith('-'):
                # continuación de palabra, quitamos el guin final
                current_line = current_line[:-1] + line
            else:
                # nueva palabra en nueva línea
                current_line += f" {line}"

        elif line.upper() == line: ## all caps
            out_lines.append(current_line)
            out_lines.append(f"## {line}")
            current_line = ""
        elif line[0].upper() == line[0]: ## first letter is upper
            out_lines.append(current_line)
            current_line = line
        else:
            print(f"\n{i:4d}>>", line)
        continue

    out_lines.append(current_line)

    total_chars_out = sum(len(line) for line in out_lines)
    print(f"output has {len(out_lines)} lines {total_chars_out} chars, skip_counts: {skip_counts}")

    return out_lines



def process_raw_lines_maestria(raw_lines: list[str]):
    """Preprocess raw text file: data/reglamento-maestria-web-2024.raw.txt
    that was obtained via online conversion of PDF file of same name
    """
    total_chars_in = sum(len(line) for line in raw_lines)
    print(f"input has {len(raw_lines)} lines, {total_chars_in} chars")

    skip_counts = Counter()
    skipped_x0c: set[str] = set()
    current_line: str = ""
    out_lines: list[str] = []

    for i, line in enumerate(raw_lines):
        if line == "":
            out_lines.append(line)
            continue

        if line.startswith("\x0c"):
            clean_line = line.replace('\x0c', '')

            if re.match("^[0-9]+$", clean_line): # just a page number?
                skip_counts["\x0c-page-number"] += 1
            # elif line not in x0c_headers: # first time seen
            #    x0c_headers.add(line)
            #    out_lines.extend([current_line, "\n"])
            #    out_lines.append(f"## {clean_line}")
            #    current_line = ""
            else: # not first time seen, skip:
                skipped_x0c.add(line)
                skip_counts["\x0c"] += 1
                print(f"skipping x0c-line: {line}")

        elif re.search("^[0-9]+\.", line):
            out_lines.extend([current_line, "\n"])
            current_line = line
        elif re.search("^[a-z]+\)", line):
            out_lines.append(current_line)
            current_line = line
        elif re.search("^\xff?Artículo +[0-9]+(.º)?$", line): # <h3>
            out_lines.extend([current_line, "\n"])
            out_lines.append(f"\n## {line}")
            current_line = ""
        elif re.search("^\xff?CAPÍTULO +[XVILCM]+", line): # <h2>
            out_lines.append(current_line)
            out_lines.append(f"\n\n# {line}")
            current_line = ""

        elif line.strip()=="UNIVERSIDAD DE LOS ANDES":
            skip_counts["UNIVERSIDAD DE LOS ANDES"] += 1
            continue
        elif re.match("^[0-9]+$", line):
            skip_counts["single-number-line"] += 1
        elif line[0].lower() == line[0]:
            if current_line.endswith('-'):
                # continuación de palabra, quitamos el guin final
                current_line = current_line[:-1] + line
            else:
                # nueva palabra en nueva línea
                current_line += f" {line}"

        elif line.upper() == line: ## all caps
            out_lines.append(current_line)
            out_lines.append(f"## {line}")
            current_line = ""
        elif line[0].upper() == line[0]: ## first letter is upper
            out_lines.append(current_line)
            current_line = line
        else:
            print(f"\n{i:4d}>>", line)
        continue

    out_lines.append(current_line)

    total_chars_out = sum(len(line) for line in out_lines)
    print(f"output has {len(out_lines)} lines {total_chars_out} chars, skip_counts: {skip_counts}")

    return out_lines


def squeeze_blank_lines(lines: list[str], n: int = 2) -> list[str]:
    """Squeeze segments of n or more consecutive blank lines into a single blank line"""

    ret = []

    for i, line in enumerate(lines):
        curr_line = line.strip()

        if curr_line != "" or i == len(lines) - 1:
            cnt = count_blank_lines_end(ret)
            if cnt >= n:
                for _ in range(cnt):
                    ret.pop()
                ret.append("")

        if curr_line != "" or i < len(lines) - 1:
            ret.append(curr_line)

    return ret


def count_blank_lines_end(a_list: list[str]) -> int:

    if len(a_list) == 0:
        return 0

    last_idx = len(a_list)-1

    for i in range(last_idx, -1, -1):
        if a_list[i] != "":
            return last_idx - i

    return last_idx + 1



def write_lines(lines: list[str], output_fpath: Path, end="\n", overwrite: bool=False) -> None:
    if output_fpath.exists() and not overwrite:
        raise FileExistsError(f"File already exists ({str(output_fpath)}) pass explicit overwrite=True to overwrite")

    with open(output_fpath, "wt", encoding="utf-8") as f_out:
        for line in lines:
            print(line, end=end, file=f_out)


# Import time Tests

assert count_blank_lines_end(["a", "", ""]) == 2
assert count_blank_lines_end(["a", "", "b"]) == 0
assert count_blank_lines_end(["", "", ""]) == 3

assert squeeze_blank_lines(["a", "", "", "b"]) == ["a", "", "b"]
assert squeeze_blank_lines(["a", "", "", "", "b"]) == ["a", "", "b"]
assert squeeze_blank_lines(["a", "", ""]) == ["a", ""]
assert squeeze_blank_lines(["a", "", "", "a", "", ""]) == ["a", "", "a", ""]
assert squeeze_blank_lines(["a", "a", "", ""]) == ["a", "a", ""]
