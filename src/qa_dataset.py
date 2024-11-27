import numpy as np
import pandas as pd
import torch as pt

from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.utils import letter_to_idx
from src.batch_enc_utils import batch_enc_apply, batch_enc_cat, pad_batch_1d_to_len

PROMPT_TMPL_VER1 = """Please answer the question following question according to the best of your knowledge of \
    *Universidad de los Andes regulations*:
{question}

Here are the answer choices:
{choices}

The correct answer is:"""


def question_formatter_ver1(row: pd.Series) -> str:
    """Genera el texto de la pregunta usando PROMPT_TMPL_VER1 que _sí_ incluye
    una lista de opciones de respuesta."""
    options = row["options"]
    assert isinstance(options, np.ndarray | list) and len(options) == 4, \
            f"options must be a list of 4 strings {options!r}"

    choices = [ f"{letter}. {options[idx]}"  for idx, letter in enumerate(["A", "B", "C", "D"])]
    return (PROMPT_TMPL_VER1.format(question=row['question'], choices="\n".join(choices)))

def answer_formatter_ver1(row: pd.Series, letter: str) -> str:
    """Retorna la respuesta correspondiente a la letra `letter` (A, B, C o D) en el formato `{letter}. {respuesta}`"""
    answer_idx = letter_to_idx(letter)
    answer_full=row["options"][answer_idx]
    return f"{letter}. {answer_full}"

FORMATTERS = {
    "ver1": (question_formatter_ver1, answer_formatter_ver1),
}


class TokenizedQAsDs(Dataset):
    """Dataset usado en loops de entrenamieno y evaluación.
    Entradas:
      - `input_df` Un dataset de preguntas  de selección múltiple, con columnas "question" y "options" al menos
         opcionalmente una columna "answer" conteniendo la letra de la respuesta correcta

      - `max_article_len`:  máxima longitud (en caracteres) de un `article` que se tendrá encuenta, p. ej. 800.
         articulos con una longitud por encima de esto simplemente se excluirán desde el principio.
      - `formatter_version`: versión de prompts usados
      - `tokenizer` Un tokenizador acorde con el modelo a evaluar o entrenar
      - `pad_to_len`: La longitud (en número de tokens) a la que se aumentan los tensores de tokens resultantes de
        concatenar tokens de entrada y de salida.
      - `pad_value`: el valor de padding que se usará para los tensores de token ids. Si no se especifica se usará
          tokenizer.pad_token_id o tokenizer.eos_token_id

      - `attention_pad_value`: el valor de padding que se usará para los tensores de atención.
         Si no se especifica se usará 0.
    """
    def __init__(self, input_df: pd.DataFrame, *,
                 formatter_version: str,
                 tokenizer: AutoTokenizer,
                 pad_to_len: int,
                 pad_value: int | None = None,
                 attention_pad_value: int = 0
                 ) -> None:

        # Basic params
        # self.max_article_len = max_article_len
        self.pad_to_len = pad_to_len
        self.attention_pad_value = attention_pad_value
        self.pad_value = pad_value

        if self.pad_value is None:
            self.pad_value = tokenizer.pad_token_id or tokenizer.eos_token_id

        self.df = self.preprocess_input(
            input_df,
            tokenizer,
            formatter_version
        )

        self.item_fields = [# "question_tokens",
            "question_len",
            "qa_tokens_A", "qa_tokens_B",  "qa_tokens_C",  "qa_tokens_D",
            "answer_mask_A", "answer_mask_B",  "answer_mask_C",  "answer_mask_D",
        ]

        if "answer" in self.df.columns:
            self.item_fields.extend([
                "correct_answer_idx",
                "question_right_ans_toks"
            ])

        if "ID" in self.df.columns:
            self.item_fields.append("ID")

        self.items: list[dict[str, pt.Tensor]] = [
            row[self.item_fields].to_dict()
            for _, row in self.df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self.items[idx]

    # Funciones auxiliares para preprocesamiento del datast original y tokenización
    def preprocess_input(
          self, input_df: pd.DataFrame,
          # max_article_len: int,
          tokenizer: AutoTokenizer,
          formatter_version: str
        ) -> pd.DataFrame:
        """Función que preprocesa el dataset crudo que se pasa como input_df (al constructor)
        Agrega las siguientes columnas:

          - `question_formatted`: el resultado de usar el question_formatter especificado por `self.formatter_version`
          - `question_tokens`: los tokens resultantes de pasar `question_formatted` por el tokenizador,
             _excluyendo_ el token EOS final
          - `question_len`:  el número de tokens en `question_tokens`
          - `correct_answer_idx`: el índice de la respuesta correcta en `options` según el mapeo natural: {
                A => 0, B => 1, C => 2, D => 3 }
          En los siguientes, la "variable" X itera sobre el conjunto {A, B, C, D}.
          - `answer_formatted_{X}: el resultado de usar el answer_formatter especificado por `self.formatter_version`
           para cada opción de respuesta
          - `answer_tokens_{X}`:  los tokens resultantes de pasar por el tokenizador.
          - `answer_len_{X}`: el número de tokens en `answer_tokens_{X}`.


          - `qa_tokens_{X}`: el resultado de concatenar `question_tokens` y `answer_tokens_{X}`.
          ** Este es el campo más importante a la hora de evaluar el desempeño de un modelo

          - `answer_mask_{X}`: este es un vector binario de la misma longitud que el almacenado en `qa_tokens_{X}`
          con 1's en las posiciones correspondientes a los tokens que pertenecen a la respuesta y 0 en el resto.
          La utilidad principal de este vector es en el proceso de evaluación porqe  facilita el cálculo vectorizado
          de la suma de los logits predichos por el modelo para cada respuesta.

          - `question_right_ans_toks`: el resultado de concatenar `question_tokens` con `answer_tokens_{R}` donde R
          es denota la respuesta correcta para la pregunta.
          ** Este es el campo más importante para el proceso de entrenamiento
        """
        # max_len_mask = (input_df['article'].str.len() < max_article_len)

        print(f"Preprocessing input dataframe  (shape={input_df.shape}) ")
              # f" # of rows with len(article) < {max_article_len} => {max_len_mask.sum()}")

        df = input_df
        # df = input_df[max_len_mask].copy()

        question_formatter, answer_formatter = FORMATTERS[formatter_version]

        df['question_formatted'] = df.apply(question_formatter, axis=1)

        df['question_tokens'] = df['question_formatted'].apply(
            lambda txt: batch_enc_apply(
                tokenizer(txt, return_tensors='pt'),
                lambda x: x.squeeze()[:-1]  # exclude EOS token, at the end
            )
        )
        df['question_len'] = df['question_tokens'].apply(
            lambda x: pt.IntTensor([x['input_ids'].shape[0]])
        )

        if 'answer' in df.columns:
            df['correct_answer_idx'] = df['answer'].apply(
                lambda let: pt.IntTensor([letter_to_idx(let)])
            )

        for let in ['A', 'B', 'C', 'D']:
              df[f'answer_formatted_{let}'] = df.apply(lambda row: answer_formatter(row, let), axis=1)

              df[f'answer_tokens_{let}'] = df[f'answer_formatted_{let}'].apply(
                  lambda txt: batch_enc_apply(
                      tokenizer(txt, return_tensors='pt'),
                      lambda x: x.squeeze()[1:] # skip first token which is SOS
                  )
              )
              df[f'answer_len_{let}'] = df[f'answer_tokens_{let}'].apply(
                  lambda x: pt.IntTensor([x['input_ids'].shape[0]]),
              )
              df[f'qa_tokens_{let}'] = df.apply(
                  lambda row: self.concat_pad_qa_tokens(row, let), axis=1
              )
              df[f'answer_mask_{let}'] = df.apply(
                  lambda row: self.make_answer_mask(row, let), axis=1
              )

        if 'answer' in df.columns:
            df['question_right_ans_toks'] = df.apply(
                lambda row: self.concat_pad_qa_tokens(row, row['answer']),
                axis=1
            )

        return df

    def make_answer_mask(self, row: pd.Series, let: str) -> pt.IntTensor:
          ret = pt.zeros(self.pad_to_len, dtype=pt.int32)
          answer_starts = row['question_len']
          answer_ends = answer_starts + row[f'answer_len_{let}']
          ret[answer_starts:answer_ends] = 1
          return ret

    def concat_pad_qa_tokens(self, row: pd.Series, let: str, ) -> pt.IntTensor:
        return pad_batch_1d_to_len(
            batch_enc_cat(row['question_tokens'], row[f'answer_tokens_{let}'], dim=0),
            target_len=self.pad_to_len,
            pad_value=self.pad_value,
            attention_pad_value=self.attention_pad_value
       )
