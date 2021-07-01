"""Create a FAISS index based on the semantic representation of an existing text corpus. To do
   so, the text will be embedded by means of a sent2vec model or a sentence-transformers model.
   The index is (basically) an efficient list that contains all the representations of the
   training corpus sentences (the TM). as such, this index can later be used to find those
   entries that are most similar to a given representation of a sentence. The index is saved to
   a binary file so that it can be reused later on to calculate cosine similarity scores and
   to retrieve the most resembling entries."""

import logging

from nfr.utils import FAISS_AVAILABLE, SENT2VEC_AVAILABLE, STRANSFORMERS_AVAILABLE, get_n_lines
from tqdm import tqdm


if FAISS_AVAILABLE:
    import faiss

if SENT2VEC_AVAILABLE or STRANSFORMERS_AVAILABLE:
    import numpy as np

if SENT2VEC_AVAILABLE:
    import sent2vec

if STRANSFORMERS_AVAILABLE:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("nfr")


def create_index(
    corpus_f: str,
    model_name_or_path: str,
    output_f: str,
    mode: str = "sent2vec",
    batch_size: int = 64,
    use_cuda: bool = False,
):
    """Given a corpus file `corpus_f` and a sent2vec model `sent2vec_f`, convert the sentences in
       the corpus (line-by-line) to vector representations, normalise them (L2norm), and add them
       to a Flat FAISS index. Finally, save the index to `output_f`.
    :param corpus_f: path to the corpus file, with one sentence per line
    :param model_name_or_path: path to the binary sent2vec model (when mode=="sent2vec") or model name of the stransformer to use
    :param output_f: path to save the FAISS index to
    :param mode: whether to use "sent2vec" or "stransformers" (sentence-transformers)
    :param batch_size: batch_size to use to create sent2vec embeddings or sentence-transformers embeddings
    :param use_cuda: whether to use GPU when using sentence-transformers
    :return: the created FAISS index
    """
    if not FAISS_AVAILABLE:
        raise ImportError(
            "Faiss not installed. Please install the right version before continuing. If you have a "
            "CUDA-enabled device and want to use GPU acceleration, you can `pip install faiss-gpu`."
            " Otherwise, install faiss-cpu. For more, see https://github.com/facebookresearch/faiss"
        )

    if mode == "sent2vec":
        if not SENT2VEC_AVAILABLE:
            raise ImportError(
                "Requested 'sent2vec', but module not installed. Install the right version from"
                " https://github.com/epfml/sent2vec"
            )
        try:
            model = sent2vec.Sent2vecModel()
        except AttributeError as exc:
            raise AttributeError(
                "'sent2vec' does not have attribute Sent2vecModel. You may have uninstalled an"
                " incorrect version of sent2vec. The correct version can be found here:"
                " https://github.com/epfml/sent2vec"
            ) from exc
        logger.info(f"Loading sent2vec model of {model_name_or_path}")
        model.load_model(model_name_or_path, inference_mode=True)
        hidden_size = model.get_emb_size()
    elif mode == "stransformers":
        if not STRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Requested 'stransformers', but module not installed. Please install the library"
                " before continuing. https://github.com/UKPLab/sentence-transformers#installation"
            )
        logger.info(f"Loading SentenceTransformer model {model_name_or_path}")
        model = SentenceTransformer(model_name_or_path, device="cuda" if use_cuda else "cpu")
        hidden_size = model.encode(["This is a test ."]).shape[1]
    else:
        raise ValueError("'mode' must be 'sent2vec' or 'stransformers'")

    logger.info(f"Creating empty index with hidden_size {hidden_size:,}...")
    # We want to do cosine similarity search, so we use inner product as suggested here:
    # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
    index = faiss.index_factory(hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)

    vecs = []
    n_lines = get_n_lines(corpus_f)
    logger.info("Converting corpus into vectors. This can take a while...")
    batch = []
    with open(corpus_f, encoding="utf-8") as fhin:
        for line_idx, line in tqdm(enumerate(fhin, 1), total=n_lines, unit="line"):
            line = line.rstrip()

            if line:
                batch.append(line)

            if len(batch) == batch_size or line_idx == n_lines:
                if mode == "sent2vec":
                    # Normalize vectors for cosine distance as suggested here:
                    # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
                    vecs.extend(model.embed_sentences(batch))
                else:
                    vecs.extend(model.encode(batch, batch_size=batch_size, show_progress_bar=False))
                batch = []

    logger.info(f"Number of entries: {len(vecs)}")

    logger.info("Normalizing vectors...")
    sent_vecs = np.array(vecs)

    # normalize_L2 works in-place so do not assign
    faiss.normalize_L2(sent_vecs)

    logger.info("Adding vectors to index...")
    index.add(sent_vecs)

    logger.info(f"Saving index to {output_f}...")
    faiss.write_index(index, output_f)

    return index


def main():
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__)
    cparser.add_argument(
        "-c",
        "--corpus_f",
        required=True,
        help="Path to the corpus to turn into vectors and add to the index. This is typically"
        " your TM or training file for an MT system containing text, one sentence per line",
    )
    cparser.add_argument(
        "-p",
        "--model_name_or_path",
        required=True,
        help="Path to sent2vec model (when `method` is sent2vec) or sentence-transformers model"
        " name when method is stransformers (see"
        " https://www.sbert.net/docs/pretrained_models.html)",
    )
    cparser.add_argument("-o", "--output_f", required=True, help="Path to the output file to write the FAISS index to")
    cparser.add_argument(
        "-m",
        "--mode",
        default="sent2vec",
        choices=["sent2vec", "stransformers"],
        help="Whether to use 'sent2vec' or 'stransformers' (sentence-transformers)",
    )
    cparser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size to use to create sent2vec embeddings or sentence-transformers"
        " embeddings. A larger value will result in faster creation, but may lead to an"
        " out-of-memory error. If you get such an error, lower the value.",
    )
    cparser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Whether to use GPU when using sentence-transformers. Requires PyTorch"
        " installation with CUDA support and a CUDA-enabled device",
    )

    cargs = cparser.parse_args()
    create_index(**vars(cargs))


if __name__ == "__main__":
    main()
