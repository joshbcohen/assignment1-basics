import regex as re
import os
from collections import defaultdict
from typing import BinaryIO

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
            split_chunks = re.split("|".join(special_tokens), chunk)
            for split_chunk in split_chunks:
                pretokens = re.finditer(PAT, split_chunk)
                for pretoken in pretokens:
                    pretoken_counts[tuple(bytes([c]) for c in pretoken[0].encode("utf-8"))] += 1
    while len(vocab) < vocab_size:
        candidate_pairs = defaultdict(int)
        for pretoken, pretoken_count in pretoken_counts.items():
            for i in range(len(pretoken) - 1):
                candidate_pairs[(pretoken[i], pretoken[i + 1])] += pretoken_count
        max_pair_key = max(candidate_pairs, key=candidate_pairs.get)
        merges.append(max_pair_key)
        vocab[len(vocab)] = b"".join(max_pair_key)
    print(pretoken_counts)
    return vocab, merges


if __name__ == "__main__":
    train_bpe("../data/TinyStoriesV2-GPT4-valid.txt", 300, special_tokens=["<|endoftext|>"])
