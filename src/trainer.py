import math
import time

from typing import Callable

import torch as pt
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader
import tqdm


class Trainer:
    def __init__(self, *,
                 train_ds: Dataset,
                 train_batch_size: int,
                 valid_ds: Dataset,
                 device: str,
                 valid_batch_size: int,
                 lr: float) -> None:
        print(f"\nInitializing trainer: device: {device} - len(train_ds)={len(train_ds)}, "
              f"len(valid_ds)={len(valid_ds)}")
        self.train_ds: Dataset = train_ds
        self.valid_ds: Dataset = valid_ds
        self.train_batch_size: int = train_batch_size
        self.valid_batch_size: int = valid_batch_size
        self.device: str = device
        self.lr: float = lr
        self.train_losses: list[float] = []
        self.valid_losses: list[float] = []
        self.best_valid_loss: float = float("inf")

    def train(self,
              model: nn.Module,
              loss_fun: Callable[[nn.Module, dict[str, Tensor]], Tensor],
              max_steps: int,
              accum_grad_steps: int,
              steps_per_log: int = 5,
              pretrain_eval: bool = False,
              ) -> nn.Module:

        opt = optim.Adam(model.parameters(), lr=self.lr)
        # n_batches = int(math.ceil(len(self.train_ds) / self.train_batch_size))
        model = model.to(self.device)

        if pretrain_eval:
            # Pre-train evaluation of the model
            # Makes sense in the case of pretrained models
            self.pretrain_eval(model, loss_fun)

        train_dl = iter(DataLoader(self.train_ds, shuffle=True,
                                   batch_size=self.train_batch_size))

        print(f'Entrenando por {max_steps} pasos.')
        print(f'Nota Importante:\n    El `Train Loss` que se reporta se calcula únicamente sobre los datos'
              f' de los últimos {accum_grad_steps} pasos de entrenamiento.\n'
              '    El `Valid Loss` es sobre *todos* los datos de validación')

        # used for logging every steps_per_log steps
        log_train_losses = []
        t0 = time.perf_counter()
        prev_token_cnt = 0
        token_cnt = 0

        for step in range(max_steps):
            model.train()

            opt.zero_grad()
            for mini_step in range(accum_grad_steps):
                batch = next(train_dl)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                token_cnt += batch['input_ids'].shape[1]

                loss = loss_fun(model, batch)
                log_train_losses.append(loss.item())
                loss /= accum_grad_steps
                loss.backward()

            opt.step()

            if (step % steps_per_log == 0 or step == max_steps-1) and self.valid_ds is not None:
                elapsed = (time.perf_counter() - t0)
                tokens_per_sec = (token_cnt - prev_token_cnt) / elapsed

                train_loss = pt.mean(pt.Tensor(log_train_losses)).item()
                pre_text = (f'Step {step:4d} - Train loss: {train_loss:5.3f}                               '
                            f'(tokens/sec:{tokens_per_sec:5.0f})')

                valid_loss = self.estimate_loss(model, loss_fun,
                                                self.valid_ds,
                                                ds_name='validation',
                                                pre_text=pre_text)

                self.train_losses.append(train_loss)
                self.valid_losses.append(valid_loss)
                # Log metrics
                print(f'Step {step:4d} -                    Valid loss: {valid_loss:5.3f}')

                # Reset counters
                t0 = time.perf_counter()
                prev_token_cnt = token_cnt
                log_train_losses = []

        return model


    @pt.no_grad
    def pretrain_eval(self, model: nn.Module,
                      loss_fun: Callable[[nn.Module, dict[str, Tensor]], Tensor]):
        model.eval()
        print("\nRunning pre-train evaluation...")
        pre_train_loss = self.estimate_loss(model, loss_fun,
                                            self.train_ds, ds_name='training')
        pre_valid_loss = self.estimate_loss(model, loss_fun,
                                            self.valid_ds, ds_name='validation')

        print(f"Pre-training eval: "
            f"train_loss={pre_train_loss:.3f} "
            f"valid_loss={pre_valid_loss:.3f}")


    @pt.no_grad
    def estimate_loss(self, model: nn.Module,
                      loss_fun: Callable[[nn.Module, dict[str, Tensor]], Tensor],
                      eval_ds: Dataset,
                      ds_name: str,
                      pre_text: str) -> float:
        model.eval()
        valid_dl = DataLoader(eval_ds, shuffle=False,
                              batch_size=self.valid_batch_size)

        n_batches = int(math.ceil(len(eval_ds) / self.valid_batch_size))
        batch_valid_losses = pt.zeros((n_batches,))

        batch_iter = tqdm.tqdm(valid_dl, desc=f"{pre_text} - estimating loss on '{ds_name}' dataset: ")
        for batch_idx, batch in enumerate(batch_iter):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # loss = model(**batch, labels=batch['input_ids']).loss
            loss = loss_fun(model, batch)
            batch_valid_losses[batch_idx] = loss.item()

        return pt.mean(batch_valid_losses).item()


