from typing import Iterable, Iterator

import regex as re

from cs336_basics.train_bpe import BYTE_CACHE, PAT_COMPILED


def pretokenized_list(text: str) -> list[tuple[bytes]]:
    pretokens = []
    byte_cache = BYTE_CACHE
    finditer = PAT_COMPILED.finditer
    for pretoken in finditer(text):
        encoded = pretoken[0].encode("utf-8")
        pretokens.append(tuple(byte_cache[b] for b in encoded))
    # print(pretokens)
    return pretokens


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_token_ids = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = sorted(special_tokens, reverse=True) if special_tokens is not None else None

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        pass

    def encode(self, text: str) -> list[int]:
        # handle special tokens
        escaped_special_tokens = map(re.escape, self.special_tokens or [])

        pattern = "(" + "|".join(escaped_special_tokens) + ")"
        encoded_bytes = []
        if self.special_tokens:
            chunks = re.split(pattern, text)
            # print(chunks)
        else:
            chunks = [text]

        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                encoded_bytes.append(self.vocab_token_ids[chunk.encode("utf-8")])  # TODO: case where not in vocab
            else:
                pretokens = pretokenized_list(chunk)
                for merge in self.merges:
                    merged_byte = b"".join(merge)
                    new_pretokens = []
                    for pretoken in pretokens:
                        new_pretoken = []
                        counter = 0
                        while counter < len(pretoken):
                            # Add new token to word at old token boundary if there's a match
                            if counter + 1 < len(pretoken) and (pretoken[counter], pretoken[counter + 1]) == merge:
                                new_pretoken.append(merged_byte)
                                counter += 2
                            else:
                                new_pretoken.append(pretoken[counter])
                                counter += 1
                        new_pretokens.append(tuple(new_pretoken))
                    pretokens = new_pretokens

                for pretoken in pretokens:
                    for token in pretoken:
                        encoded_bytes.append(self.vocab_token_ids[token])

        return encoded_bytes

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        decoded_byte_list = [self.vocab.get(id, b"") for id in ids]
        decoded_byte_str = b"".join(decoded_byte_list)
        return bytes.decode(decoded_byte_str, errors="replace")
