try:
    import sentence_transformers

    STRANSFORMERS_AVAILABLE = True
except ImportError:
    STRANSFORMERS_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import sent2vec

    SENT2VEC_AVAILABLE = True
except ImportError:
    SENT2VEC_AVAILABLE = False


def get_n_lines(fin: str, size: int = 65536) -> int:
    """Given a filename, return how many lines (i.e. line endings) it has.
    :param fin: input file
    :param size: size in bytes to use as chunks
    :return: number of lines (i.e. line endings) that `fin` has
    """
    # borrowed from https://stackoverflow.com/a/9631635/1150683
    def blocks(fh):
        while True:
            b = fh.read(size)
            if not b:
                break
            yield b

    with open(str(fin), encoding="utf-8") as fhin:
        return sum([bl.count("\n") for bl in blocks(fhin)])
