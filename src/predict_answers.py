from pathlib import Path

import pandas as pd
import torch as pt
import tqdm
from torch import nn, Tensor
from torch.utils.data import DataLoader
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
        pad_to_len=256
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