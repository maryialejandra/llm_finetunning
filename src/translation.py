import time
import src.auth_utils as au

from pathlib import Path
from groq import Groq

class GroqTranslator:
    def __init__(self, *,
                 api_keys_var: str,
                 groq_model: str,
                 pause_secs: float = 2.0,
                 ):
        groq_api_keys = au.get_secret(api_keys_var).split(";")
        print(f"{len(groq_api_keys)} Groq api keys loaded.")
        self.groq_model = groq_model
        self.pause_secs = pause_secs
        self.clients = [Groq(api_key=key) for key in groq_api_keys]
        self.client_idx = 0
        self.cache: dict[str, str] = {}

        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.queue_time: float = 0
        self.total_time: float = 0

        self.prompt_tmpl = ("Please translate the following text pertaining university regulations at Universidad de los Andes to English:\n"
                            "```\n{spanish_text}\n```\nProvide ONLY ONE translation, the best one, and DO NOT add any other content or comments besides the translation. Also, please DO NOT preface the translation with any text such as 'The translation of the text to English is' or anything similar:\n")
        # CompletionUsage(completion_tokens=60, prompt_tokens=139, total_tokens=199, completion_time=0.24, prompt_time=0.033716789, queue_time=0.19504005, total_time=0.273716789)

    def translate(self, spanish_text: str, verbose: bool = False) -> str:
        if spanish_text.strip() == "":
            return ""

        print(f"Spanish text    : {spanish_text[0:45]}...{spanish_text[-45:]} ...")
        if spanish_text in self.cache:
            print("Found in cache!")
            return self.cache[spanish_text]
        else:
            prompt = self.prompt_tmpl.format(spanish_text=spanish_text)

            if verbose:
                print(f"==========PROMPT=====\n{prompt}\n====================")

            try:
                client = self.clients[self.client_idx]
                chat_completion = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    model=self.groq_model
                )
                # print(chat_completion.usage)
                usage = chat_completion.usage
                self.prompt_tokens += usage.prompt_tokens
                self.completion_tokens += usage.completion_tokens
                self.queue_time += usage.queue_time
                self.total_time += usage.total_time

                translation_text = chat_completion.choices[0].message.content
                self.cache[spanish_text] = translation_text.strip("'").strip('"')

                time.sleep(self.pause_secs)
            except Exception as exc:
                raise exc

        print(f"English translation: {translation_text[0:45]}...{translation_text[-45:]}")

        return translation_text.strip("'")


def translate_file(file_prefix: str, translator: GroqTranslator) -> None:
    input_lines = Path(f"../data/{file_prefix}.preprocessed.txt").read_text().split("\n")
    print(f"{len(input_lines)} input lines ")
    n_lines = len(input_lines)

    out_path = Path(f"../data/{file_prefix}.translated.txt")
    print(f"Translated text will be written to: {out_path}")
    with out_path.open("wt") as f_out:
        for i, line in enumerate(input_lines):
            print(f"{i+1:4d} / {n_lines} ({100.0 * (i+1)/n_lines:.1f}%)")
            line_translation = translator.translate(line)
            print(line_translation, file=f_out, flush=True)
