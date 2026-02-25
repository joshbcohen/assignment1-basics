import regex as re
import os
from collections import defaultdict
from typing import BinaryIO
from operator import itemgetter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_CHUNKS = 4  # TODO: Parallelize pretokenizing on chunks


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _get_token_pairs(pretoken: tuple[bytes]) -> list[bytes]:
    return [(pretoken[i], pretoken[i + 1]) for i in range(len(pretoken) - 1)]

def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = "<|endoftext|>".encode("utf-8")
    pretoken_counts = defaultdict(int)
    merges = []
    with open(input_path, "rb") as f:
        chunk_boundaries = _find_chunk_boundaries(f, NUM_CHUNKS, b"<|endoftext|>")
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            split_chunks = re.split("|".join(map(re.escape, special_tokens)), chunk)
            # TODO:  Parallelize on chunk boundaries with multiprocessing
            for split_chunk in split_chunks:
                pretokens = re.finditer(PAT, split_chunk)
                for pretoken in pretokens:
                    pretoken_counts[tuple(bytes([c]) for c in pretoken[0].encode("utf-8"))] += 1
    # Create count of token pairs in corpus
    candidate_pair_counts = defaultdict(int)
    for pretoken, pretoken_count in pretoken_counts.items():
        for i in range(len(pretoken) - 1):
            candidate_pair_counts[(pretoken[i], pretoken[i + 1])] += pretoken_count
    while len(vocab) < vocab_size:
        # Get max first on pair count in corpus, then on pair itself
        # to get lexicographically greater pair
        max_pair = max(candidate_pair_counts.items(), key=itemgetter(1, 0))[0]
        merges.append(max_pair)
        new_byte = b"".join(max_pair)
        vocab[len(vocab)] = new_byte

        new_pretoken_counts = {}
        for word, freq in pretoken_counts.items():
            token_pairs = _get_token_pairs(word)
            if max_pair in token_pairs:
                # Create new word
                new_word = []
                counter = 0
                while counter < len(word):
                    # Add new token to word at old token boundary if there's a match
                    if counter + 1 < len(word) and (word[counter], word[counter + 1]) == max_pair:
                        new_word.append(new_byte)
                        counter += 2
                    else:
                        new_word.append(word[counter])
                        counter += 1
                new_word = tuple(new_word)
                new_pretoken_counts[new_word] = freq
                # Reduce frequencies of token pairs from old word
                for token_pair in token_pairs:
                    if token_pair in candidate_pair_counts:
                        candidate_pair_counts[token_pair] -= freq
                        if candidate_pair_counts[token_pair] == 0:
                            del candidate_pair_counts[token_pair]
                # Increment frequencies of token pairs from new word
                for token_pair in _get_token_pairs(new_word):
                    candidate_pair_counts[token_pair] += freq
            else:
                new_pretoken_counts[word] = freq
        pretoken_counts = new_pretoken_counts
    return vocab, merges


if __name__ == "__main__":
    # vocab, merges = train_bpe("../data/TinyStoriesV2-GPT4-valid.txt", 500, special_tokens=["<|endoftext|>"])
    # print(f"BPE tokenizer vocab: {vocab}")
    # print(f"BPE tokenizer merges: {merges}")
    vocab, merges = train_bpe("../data/corpus.en", 500, special_tokens=["<|endoftext|>"])
    # print(f"BPE tokenizer vocab: {vocab}")
    print(f"BPE tokenizer merges:")
    for t1, t2 in merges:
        print(f"{t1}, {t2}")
