import re
import time
import src.utils as ut
from groq import Groq, RateLimitError


PROMPT_TMPL_ENG_V0: str = """
Please generate one multiple choice question, a correct answer for it and three (3) incorrect answers, based on the following text:
```
{chunk}
```
The question should be short, i.e. between about 8 and 15 words long and end on question mark.
The question should not be about chapter/section/article numbers or titles.
You should return the question, followed by the correct answer, and then three incorrect answers.
The correct answer and each of the incorrect answers should be short, each no longer than a few words (10 tops.)

QUESTION: <question here between 8 and 15 words long ending in question mark>
CORRECT ANSWER: <correct answer here>
INCORRECT ANSWER 1: <incorrect answer 1 here>
INCORRECT ANSWER 2: <incorrect answer 2 here>
INCORRECT ANSWER 3: <incorrect answer 3 here>

QUESTION:"""


PROMPT_TMPL_ENG_V1: str = """
Please generate one multiple choice question, a correct answer for it and three (3) incorrect answers, based on the following text:
```
{chunk}
```
The question should be short, i.e. between about 8 and 15 words long and end on question mark, and should be contained in a SINGLE line
DO NOT preface the question with any introductory remark, except "QUESTION:"
The question should not be about chapter/section/article numbers or titles.
You should return the question in the first line, followed by the correct answer in the 2nd line.
Lines 3, 4 and 5 should contain an incorrect answer each.
The correct answer and each of the incorrect answers should be short, each no longer than a few words (10 tops.)

i.e we expect the following format:

QUESTION: <question here between 8 and 15 words long ending in question mark>
CORRECT ANSWER: <correct answer here>
INCORRECT ANSWER 1: <incorrect answer 1 here>
INCORRECT ANSWER 2: <incorrect answer 2 here>
INCORRECT ANSWER 3: <incorrect answer 3 here>

QUESTION:"""


PROMPT_TMPL_ENG_V2: str = """
Please generate one multiple choice question (with a single right answer and 3 wrong answers), based on the text given below:
The question SHOULD NOT BE about chapter numbers, chapter titles or article numbers

i.e. the expected format is as follows:

```
Question: <generated question here no more than 10 o 15 palabras and ending on a question mark>
Right answer: <right answer here>
Wrong answer 1: <wrong answer 1 here>
Wrong answer 2: <wrong answer 2 here>
Wrong answer 3: <wrong answer 3 here>


The text about which you should generate generate the question is:
```
{chunk}
```

Question:"""



PROMPT_TMPL_SPA_V0: str = """
Por favor genera una pregunta de opción múltiple (con una única respuesta correcta y 3 respuestas incorrectas) a partir del fragmento de texto que te doy más abajo.
La pregunta NO DEBE ser sobre números o títulos de artículos o capítulos.

Es decir, el formato que esperamos es:

```
Pregunta: <pregunta aquí, no más de 10 o 15 palabras y termina en signo de interrogación>
Respuesta correcta: <respuesta correcta aquí>
Respuesta incorrecta 1: <respuesta incorrecta 1 aquí>
Respuesta incorrecta 2: <respuesta incorrecta 2 aquí>
Respuesta incorrecta 3: <respuesta incorrecta 3 aquí>
```

El texto a partir del cuál debes generar la pregunta es:
```
{chunk}
```
Pregunta: """



PROMPT_TMPL_SPA_V1: str = """
Por favor genera una pregunta de opción múltiple (con una única respuesta correcta y 3 respuestas incorrectas) a partir del fragmento del siguiente texto:
La pregunta NO DEBE ser sobre números o títulos de artículos o capítulos.

```
{chunk}
```

Instruccciones:
1. Genera primero una única línea con la pregunta.
2. También genera una sola línea con la respuesta correcta.
3. Además genera otras tres (3) líneas con respuestas incorrectas.

Es decir, el formato que esperamos es:

Pregunta: <pregunta aquí, no más de 10 o 15 palabras y termina en signo de interrogación>
Respuesta correcta: <respuesta correcta aquí>
Respuesta incorrecta 1: <respuesta incorrecta 1 aquí>
Respuesta incorrecta 2: <respuesta incorrecta 2 aquí>
Respuesta incorrecta 3: <respuesta incorrecta 3 aquí>

Pregunta:"""


MATCH_STRS_ENG: dict[str, str] = {
    "question": "QUESTION",
    "correct": "CORRECT ANSWER",
    "incorrect": "INCORRECT ANSWER"
}

MATCH_STRS_ENG_V2: dict[str, str] = {
    "question": "Question",
    "correct": "Right answer",
    "incorrect": "Wrong answer"
}

MATCH_STRS_SPA: dict[str, str] = {
   "question": "Pregunta",
   "correct": "Respuesta correcta",
   "incorrect": "Respuesta incorrecta"
}

PROMPTS_BY_VER: dict[str, tuple[str, dict[str, str]]] = {
    "eng-v0": (PROMPT_TMPL_ENG_V0, MATCH_STRS_ENG),
    "eng-v1": (PROMPT_TMPL_ENG_V1, MATCH_STRS_ENG),
    "eng-v2": (PROMPT_TMPL_ENG_V2, MATCH_STRS_ENG_V2),
    "spa-v0": (PROMPT_TMPL_SPA_V0, MATCH_STRS_SPA),
    "spa-v1": (PROMPT_TMPL_SPA_V1, MATCH_STRS_SPA),
}


class QAGenerator:
    def __init__(self, *,
                 api_keys_var: str,
                 groq_model: str,
                 prompt_ver: str,
                 pause_secs: float = 2.0,
                 cache_enabled: bool = True,
                 temperature: float = 0.0,
                 ):
        groq_api_keys = ut.get_secret(api_keys_var).split(";")
        print(f"{len(groq_api_keys)} Groq api keys loaded.")
        self.groq_model = groq_model
        self.temperature = temperature
        self.pause_secs = pause_secs
        self.clients = [Groq(api_key=key) for key in groq_api_keys]
        self.client_idx = 0
        self.cache_enabled = cache_enabled
        self.cache: dict[str, str] = {}

        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.queue_time: float = 0
        self.total_time: float = 0

        self.prompt_ver = prompt_ver
        self.prompt_tmpl = PROMPTS_BY_VER[prompt_ver][0]
        self.match_strs = PROMPTS_BY_VER[prompt_ver][1]
        print(f"QAGenerator.init: prompt_tmpl=\n'''{self.prompt_tmpl}'''")

    def gen_question(self, chunk: str, verbose: bool = False) -> str:
        assert chunk.strip() != ""

        if self.cache_enabled and chunk in self.cache:
            print("Found in cache!")
            return self.cache[chunk]
        else:
            prompt = self.prompt_tmpl.format(chunk=chunk)

            if verbose:
                print(f"==========QA GENERATION PROMPT=====\n{prompt}\n====================\n")

            while True:
                try:
                    client = self.clients[self.client_idx]
                    chat_completion = client.chat.completions.create(
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }],
                        model=self.groq_model,
                        temperature=self.temperature
                    )
                    usage = chat_completion.usage
                    self.prompt_tokens += usage.prompt_tokens
                    self.completion_tokens += usage.completion_tokens
                    self.queue_time += usage.queue_time
                    self.total_time += usage.total_time

                    generated_qa = chat_completion.choices[0].message.content

                    lines = [ln for ln in generated_qa.split("\n") if ln.strip() != ""]
                    if len(lines) != 5 or self.match_strs["correct"] not in generated_qa:
                        print(f"WARNING: invalid generated_qa: {generated_qa!r}")
                        time.sleep(self.pause_secs)
                        continue # Retry

                    if self.cache_enabled:
                        self.cache[chunk] = generated_qa.strip("'").strip('"')

                    time.sleep(self.pause_secs)
                    if verbose:
                        print(f"=== GENERATED QUESTION AND ANSWERS ====\n{generated_qa}\n=========================\n")
                    return generated_qa.strip("'")

                except RateLimitError as rlerr:
                    self.client_idx = (self.client_idx + 1) % len(self.clients)
                    print(f"Rate limit error: {rlerr} retrying with new client: {self.client_idx}")


def parse_generated_question(generated_qa: str, match_strs: dict[str, str]) -> dict[str, str]:
    # Remove empty lines
    lines = [ln for ln in generated_qa.split("\n") if ln.strip() != ""]

    if len(lines) == 5:
        question = re.sub(match_strs["question"]  + ": *", "", lines[0])
        correct_answer_idx = 1
        wrong_answer_idx = 2
    else:
        correct_answer_idx = ut.find_first_idx(lines, lambda x: match_strs["correct"] in x)
        assert correct_answer_idx is not None, f"generated_qa: {generated_qa!r}"

        question_lines = [ln.replace(match_strs["question"], "").strip()
                          for ln in lines[:correct_answer_idx]]
        question = " ".join(question_lines)

        wrong_answer_idx = ut.find_first_idx(lines, lambda x: match_strs["incorrect"] in x)

    assert wrong_answer_idx is not None, f"generated_qa: {generated_qa!r}"

    correct_answer_lines =  [re.sub(" *" + match_strs["correct"]+ " *:?", "", ln )
                             for ln in lines[correct_answer_idx:wrong_answer_idx]]

    correct_answer = " ".join(correct_answer_lines)

    ret = {
        "question": question,
        "correct_answer": correct_answer,
        "incorrect_answers": []
    }

    for i in range(3):
        answer = lines[wrong_answer_idx + i]
        if match_strs["incorrect"] in answer:
            clean_answer = re.sub(" *" + match_strs["incorrect"] +" *[0-9]+:?", "", answer)
            ret["incorrect_answers"].append(clean_answer.strip())

    return ret



def parse_generated_question_v0(generated_qa: str):
    # Remove empty lines
    lines = [ln for ln in generated_qa.split("\n") if ln.strip() != ""]

    if len(lines) != 5:
        print("WARNING: Something possibly wrong with this generated_qa "
              f"(does not have 5 lines, but {len(lines)}): {lines!r}")

    question = lines[0]
    if question.startswith("QUESTION:"):
        question = question[9:].strip()

    correct_answer = lines[1]
    if correct_answer.startswith("CORRECT ANSWER:"):
        correct_answer = correct_answer[len("CORRECT ANSWER:"):].strip()

    ret = {
        "question": question,
        "correct_answer": correct_answer,
        "incorrect_answers": []
    }

    for i in range(3):
        answer = lines[2 + i]
        if answer.startswith("INCORRECT ANSWER"):
            ret["incorrect_answers"].append(answer[len("INCORRECT ANSWER") + 3:].strip())

    return ret


def enrich_generated_qa(in_rec: dict[str, str | list[str]],
                        src_chunk: str, example_id: str):
    """Add the following fields to rec:
       - example_id: str - derived from src_file_key
       - answer:  Literal["A","B","C","D"]; letter corresponding to right answer
       - options: list[str]; all answer options including correct answer in position
                  corresponding to the value of key 'answer'
    """

    ret = {}
    ret["source_chunk"] = src_chunk
    ret["question"] = in_rec["question"]
    ret["correct_answer"] = in_rec["correct_answer"]
    ret["example_id"] = example_id

    perm_idxs = ut.random_permutation([0, 1, 2, 3])
    # Will put correct answer in options at this index:
    answer_idx = perm_idxs[0]
    ret["answer"] = "ABCD"[answer_idx]

    options = [None] * 4
    options[answer_idx] = in_rec["correct_answer"]
    # Use remaining indices to put incorrect answers

    # assert len(in_rec["incorrect_answers"])==3, f"in_rec={in_rec!r}"

    wrong_answers = ensure_incorrect_answers(in_rec["incorrect_answers"])
    for idx, incorrect in zip(perm_idxs[1:], wrong_answers[:3],
                              strict=True):
        options[idx] = incorrect.strip()

    ret["options"] = options

    return ret


def ensure_incorrect_answers(answers: list[str]) -> list[str]:
    """Ensure that answer has exactly 3 incorrect answers"""

    ret = answers.copy()

    while len(ret) < 3:
        ret.append("The answer is 42")

    return ret
