from typing import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import torch as pt
from tqdm import tqdm
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

import src.utils as ut
from src.qa_dataset import TokenizedQAsDs
from src.utils import module_device, letter_to_idx



def gen_test_predictions_with_model(
            model: nn.Module,
            formatter_version: str = "ver1",
            batch_size: int = 4,
            out_csv_path: Path | None = None
        ) -> pd.DataFrame:
    ut.login_to_hf_hub()

    test_df = ut.load_test_df()
    tokenizer = AutoTokenizer.from_pretrained(ut.LLAMA_MODEL_ID)

    test_ds = TokenizedQAsDs(
        test_df,
        formatter_version=formatter_version,
        tokenizer=tokenizer,
        pad_to_len=256,
        device=None
    )

    return predict_all(model, test_ds,
                       batch_size=batch_size,
                       out_csv_path=out_csv_path)


def predict_all(model: nn.Module,
                qa_ds: TokenizedQAsDs,
                batch_size: int = 4,
                out_csv_path: Path | None = None
                ) -> pd.DataFrame:

    test_dl = DataLoader(qa_ds, batch_size=batch_size, shuffle=False)

    answers = []
    for batch in tqdm.tqdm(test_dl):
        ans = predict_answer_batch(model, batch)
        answers.append(ans)

    ids = pt.cat([dic["ID"] for dic in answers]).numpy()
    best_answer = pt.cat([dic["best_answer"] for dic in answers]).numpy()

    answers_df = pd.DataFrame(
        data={
            "ID": ids,
            "answer": best_answer + 1  # produce 1-based answers, i.e in {1, 2, 3, 4}
        }
    )

    if out_csv_path is not None:
        answers_df.to_csv(out_csv_path, index=False)

    return answers_df


def predict_answer_batch(
        model: nn.Module,
        batch: dict[str, Tensor|BatchEncoding],
        logit_aggr: str = "mean",
        shift_ids: bool = True,
        verbose: bool = False
      ) -> dict[str, Tensor]:

    device = module_device(model)
    model.eval()

    ret = {}

    if "ID" in batch:
        ret["ID"] = batch["ID"].squeeze()

    if "correct_answer_idx" in batch:
        ret["correct_answer_idx"] = batch["correct_answer_idx"].squeeze()

    batch_sz = batch['qa_tokens_A']['input_ids'].shape[0]
    ret['logits_by_let'] = pt.zeros([4, batch_sz])

    if verbose:
        def log(*x): print(*x)
    else:
        def log(*_): pass

    with pt.no_grad():
        for let in ['A', 'B', 'C', 'D']:
            # print(f"input_ids.shape (before unsqueeze)={instance[f'qa_tokens_{let}']['input_ids'].shape}")
            input_ids: Tensor = batch[f'qa_tokens_{let}']['input_ids']
            log("input_ids.shape", input_ids.shape)
            assert len(input_ids.shape) == 2
            # assert input_ids.shape[0] == 1

            # Get model outputs
            outputs = model(
                input_ids.to(device),
                use_cache=True,
                return_dict=True
            )

            # dev1 = 'cpu'
            dev1 = device

            logits = outputs.logits.to(dev1)  # shape: [batch_sz, seq_len, vocab_size]
            log("\nlogits.shape:", logits.shape)

            # NEED TO SHIFT BY ONE HERE AS logit[b, 0, :] is prediction of input_id[b, 1]
            if shift_ids:
                logits = logits[:, :-1, :]  # shape: [batch_sz, seq_len-1, vocab_size]

            answer_mask = batch[f'answer_mask_{let}'].to(dev1)  # [batch_sz, seq_len]

            y_ids = input_ids.to(dev1).unsqueeze(-1) # shape: [batch_sz, seq_len, 1]
            if shift_ids:
                y_ids = y_ids[:, 1:, :]
                answer_mask = answer_mask[:, 1:] # shape: [batch_sz, seq_len - 1]

            log("\ny_ids.shape=", y_ids.shape)
            log("\nanswer_mask.shape", answer_mask.shape)

            # gathered_logits[b, i] := logits[b][i][idx[b][i][0]]  # i in range(seq_len - 1)
            gathered_logits = logits.gather(dim=2, index=y_ids).squeeze()  # [batch_sz, seq_len(-1)]

            log("\ngathered_logits.shape", gathered_logits.shape)
            answer_range_logits = gathered_logits * answer_mask

            if logit_aggr == "mean":
                answer_len = answer_mask.sum(dim=1) # shape: [batch_sz, 1]
                aggregated_logit = (answer_range_logits.sum(dim=1) / answer_len)
            elif logit_aggr == "sum": # logit_aggr == "max"
                aggregated_logit = answer_range_logits.sum(dim=1)
            else:
                raise ValueError(f"Invalid logit aggregation method: {logit_aggr}")

            ret['logits_by_let'][letter_to_idx(let), :] = aggregated_logit.to(pt.float32)

        # end-for let
        ret["best_answer"] = ret['logits_by_let'].argmax(dim=0)

        if "correct_answer_idx" in batch:
            ret["is_correct"] = ret["best_answer"] == ret["correct_answer_idx"]

        return ret


def predict_best_answer_v1(model, tokenizer: AutoTokenizer,
                        example: dict[str, str],
                        question_formatter,
                        logit_aggr: str = "sum",  # or "mean"
                        verbose = True):
  """Buggy do not use!"""
  model.eval()  # Set model to evaluation mode
  device = ut.module_device(model)

  with pt.no_grad():
        if not isinstance(example, dict | pd.Series):
            raise TypeError(f"Expected a dictionary or series, but got {type(example)}: {example}")

        # Prepare the input
        input_text = question_formatter(example)

        # Tokenize the input
        input_ids = tokenizer(input_text, return_tensors="pt",
                              truncation=True,
                              # max_length=800
                              ).input_ids.to(device)
        # print(f"input_ids={input_ids[0, :5]}..{input_ids[0, -5:]}")

        # Generate logits
        # if verbose:
        # print(f"len(input_text) = {len(input_text)}")
        # print(f"input_ids.shape = {input_ids.shape}")

        # probabilities = softmax(logits, dim=-1)

        # Compute scores for each option
        option_scores = []
        logits = model(input_ids).logits[0]

        for i in [1, 2, 3, 4]:
            option = example[f"Opcion{i}"]

            option_ids = tokenizer(option,
                                  return_tensors="pt",
                                  add_special_tokens=False
                                  ).input_ids.to(device)

            # model_input = pt.cat([input_ids, option_ids], dim=1)
            # print(f"\n\nmodel_input.shape = {model_input.shape}")

            # print(f"logits.shape = {logits.shape}")
            # print(f"option {i}: len: {len(option)} option_ids.shape = {option_ids.shape}")
            probs = pt.nn.functional.softmax(logits, dim=-1)
            # print(f"probs: {probs.shape} {probs.sum(axis=1)}")

            option_score = 0.
            n_opt_tokens = option_ids.shape[1]
            # print("n_opt_tokens", n_opt_tokens)
            for ans_idx in range(n_opt_tokens):
                option_score += probs[ans_idx, option_ids[0][ans_idx]].item()

            if logit_aggr == "mean":
                option_score /= n_opt_tokens
            elif logit_aggr == "sum":
                pass
            else:
                raise ValueError(f"Unknown prob_agg: {logit_aggr}")

            # print("question", example["Pregunta"],  i, option, option_score)
            option_scores.append(option_score)

        # Find the index of the highest probability option
        return np.argmax(option_scores)


def answer_all_and_save(test_df: pd.DataFrame,
                        model: nn.Module,
                        model_desc: str,
                        tokenizer: AutoTokenizer,
                        logit_aggr: str,
                        question_formatter
                        ) -> pd.DataFrame:
    answers = {}
    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       desc="predicting answers for test_df"):
        # answer here is one of 0, 1, 2, 3
        answers[ row['ID'] ] = predict_best_answer_v1(model, tokenizer, row,
                                                   logit_aggr=logit_aggr,
                                                   question_formatter=question_formatter)

    # make it one of 1, 2, 3, 4
    df = pd.DataFrame(pd.Series(answers) + 1).reset_index()
    df.columns = ["ID", "Respuesta"]

    fmt_name = question_formatter.__name__
    fname = f"{model_desc}_{logit_aggr}_{fmt_name}.csv".strip('.').strip('-')
    out_fpath = ut.DATA_PATH / fname
    print("output saved to:", out_fpath )
    df.to_csv(out_fpath, index=False)

    return df


def get_highest_probability_option(model, question_formatter, tokenizer, example):

    model.eval()  # Set model to evaluation mode
    device = ut.module_device(model)

    with pt.no_grad():
            if not isinstance(example, dict | pd.Series):
                raise TypeError(f"Expected a dictionary or Series, but got {type(example)}: {example}")

            input_text = question_formatter(example)
            # Tokenize the input
            input_ids = tokenizer(input_text, return_tensors="pt",
                                  truncation=True, max_length=800).input_ids.to(device)

            # Generate logits
            logits = model(input_ids).logits
            probabilities = softmax(logits, dim=-1)

            # Compute scores for each option
            option_scores = []
            for i, option in enumerate([
                example["Opcion1"],
                example["Opcion2"],
                example["Opcion3"],
                example["Opcion4"]
            ]):
                option_ids = tokenizer(option, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                option_score = probabilities[0, :, option_ids].sum().item()  # Sum of probabilities
                option_scores.append(option_score)

            # Find the index of the highest probability option
            highest_prob_index = np.argmax(option_scores)
            # highest_prob_indices.append(highest_prob_index)  # Append the index (0 for Opcion1, etc.)

    return highest_prob_index

@dataclass
class PredictorArgs:
    model: nn.Module
    model_desc: str
    tokenizer: AutoTokenizer
    question_formatter: Callable[[dict[str, str]], str]
    option_formatter: Callable[[dict[str, str], int], str]
    logit_aggr: str = "sum"  # or "mean"
    max_len: int = 216
    verbose = True


def predict_best_answers_v0(test_df: pd.DataFrame,
                           args: PredictorArgs) -> tuple[pd.DataFrame, Path]:
    answers = {}
    for _, row in tqdm(test_df.iterrows()):
        answers[ row['ID'] ] = get_highest_probability_option(
            args.model,
            args.question_formatter,
            args.tokenizer,
            row.to_dict()
        )

    df = pd.DataFrame(pd.Series(answers) + 1).reset_index()
    df.columns = ["ID", "Respuesta"]

    fmt_name = args.question_formatter.__name__
    fname = f"{args.model_desc}_{fmt_name}-v0.csv".strip('.').strip('-')
    out_fpath = ut.DATA_PATH / fname
    print("output saved to:", out_fpath )
    df.to_csv(out_fpath, index=False)

    return df, out_fpath



def predict_best_answer(args: PredictorArgs, example: dict[str, str]) -> int:
    """Return best option index for this example ()"""
    args.model.eval()  # Set model to evaluation mode
    device = ut.module_device(args.model)

    assert isinstance(example, dict | pd.Series), \
        f"Expected a dictionary or series, but got {type(example)}: {example}"

    q_text = args.question_formatter(example)
    q_ids = args.tokenizer(q_text, return_tensors="pt").input_ids.squeeze().to(device)  # shape: (q_ids_len,)
    q_ids_len = q_ids.shape[0]

    input_idss: list[Tensor] = []
    mask_idss: list[Tensor] = []

    eos_id = args.tokenizer.eos_token_id

    # prepare inputs to model:
    for i in range(4):
        option_text = args.option_formatter(example, i)
        option_ids = (args.tokenizer(
                              option_text,
                              return_tensors="pt",
                              add_special_tokens=False
                      )
                      .input_ids.squeeze()
                      .to(device))

        # print(tokenizer.decode(option_ids))
        opt_ids_len = option_ids.shape[0]

        assert q_ids_len + opt_ids_len <= args.max_len
        remainder = args.max_len - (q_ids_len + opt_ids_len)

        input_ids = pt.cat([q_ids, option_ids,
                            eos_id * pt.ones(remainder, dtype=pt.int64).to(device)
                            ])
        input_idss.append(input_ids)

        # put ones in places corresponding to the option
        mask_ids = pt.cat([pt.zeros(q_ids_len),
                          pt.ones(opt_ids_len),
                          pt.zeros(remainder)]
                          ).to(device)

        mask_idss.append(mask_ids)
        # print('input_ids: ', input_ids)
        # print('opt_ids', option_ids)
        # print('input_ids', input_ids)
        # print('mask_ids', mask_ids)
        # print('input * mask', input_ids * mask_ids)

    input_ids_stack = pt.stack(input_idss)
    mask_ids_stack = pt.stack(mask_idss)

    # predict:
    outputs = args.model(
        input_ids_stack.to(device),
        use_cache=True,
        return_dict=True
    )
    #print("output_logits.shape:  ", outputs.logits.shape)

    # Gather logits:
    # NEED TO SHIFT BY ONE both input_ids and masks HERE
    # becaue logit[0] is prediction of input_id[1]

    logits = outputs.logits[:, :-1, :]  # shape: [4, max_len - 1, vocab_size]
    #print("logits-1.shape:       ", logits.shape)

    idx = input_ids_stack[:, 1:].unsqueeze(-1) # shape: [4, max_len -1, 1]
    # print("idx.shape:            ", idx.shape)

    gathered_logits = logits.gather(dim=2, index=idx).squeeze()
    # print(f"gathered_logits.shape={gathered_logits.shape}")
    opts_logits = gathered_logits * mask_ids_stack[:, 1:]
    if args.logit_aggr == "sum":
        logits_aggred = opts_logits.sum(dim=1)
    elif args.logit_aggr == "mean":
        logits_aggred = opts_logits.sum(dim=1) / mask_ids_stack.sum(dim=1)
    else:
        raise ValueError(f"logit_aggr must be 'sum' or 'mean', but got {args.logit_aggr}")

    best_idx = pt.argmax(logits_aggred).item()  # 0-based!
    # print(f"best_idx: {best_idx}")
    return best_idx


def predict_best_answers_all(args: PredictorArgs, test_df: pd.DataFrame) \
    -> tuple[pd.DataFrame, Path]:
    answers = {}
    for _, row in tqdm(test_df.iterrows()):
        # answer here is one of 0, 1, 2, 3
        answers[row['ID']] = predict_best_answer(args, row.to_dict())

    # make it one of 1, 2, 3, 4
    df = pd.DataFrame(pd.Series(answers) + 1).reset_index()
    df.columns = ["ID", "Respuesta"]

    fmt_name = args.question_formatter.__name__
    out_fpath = ut.DATA_PATH / f"{args.model_desc}-{args.logit_aggr}-{fmt_name}.csv"
    print("output saved to:", out_fpath )
    df.to_csv(out_fpath, index=False)

    return df, out_fpath
