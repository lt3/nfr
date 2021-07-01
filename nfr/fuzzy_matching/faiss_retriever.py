"""Find fuzzy matches based on semantic similarity. This similarity is provided by a FAISS index
   which holds vector representations of the TM's source side. We convert a given sentence into
   a vector representation with sent2vec, and look for the most similar (cosine similarity)
   topk matches. When using the class directly, the source and target TM can be text files or
   lists of sentences."""

import linecache
import logging
from typing import List, Optional, Union

from nfr.utils import FAISS_AVAILABLE, SENT2VEC_AVAILABLE, STRANSFORMERS_AVAILABLE


if FAISS_AVAILABLE:
    import faiss

if SENT2VEC_AVAILABLE or STRANSFORMERS_AVAILABLE:
    import numpy as np

if SENT2VEC_AVAILABLE:
    import sent2vec

if STRANSFORMERS_AVAILABLE:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("nfr")


class FaissRetriever:
    def __init__(
        self,
        corpus_src: Union[str, List[str]],
        corpus_tgt: Union[str, List[str]],
        model_name_or_path: str,
        faiss_f,
        mode: str = "sent2vec",
        use_cuda: bool = False,
    ):
        """Initialize the FaissRetriever. Peculiarity: `corpus_src` and `corpus_tgt` can
           either be a filename or a list of sentences.
        :param corpus_src: file name or list of sentences of the source side of the TM (which is contained as vectors
               in the FAISS index)
        :param corpus_tgt: file name or list of sentences of the target side of the TM
        :param model_name_or_path: path to the binary sent2vec model (when mode=="sent2vec") or model name of the stransformer to use
        :param faiss_f: Path to the FAISS index
        :param mode: whether to use "sent2vec" or "stransformers" (sentence-transformers)
        :param use_cuda: whether or not to use CUDA for the FAISS index and sentence-transformers.
               This should NOT be run in multiprocessing context
        """
        self.corpus_src = corpus_src
        self.corpus_tgt = corpus_tgt
        self.model_name_or_path = model_name_or_path
        self.mode = mode
        self.use_cuda = use_cuda

        if not FAISS_AVAILABLE:
            raise ImportError(
                "Faiss not installed. Please install the right version before continuing. If you have a "
                "CUDA-enabled device and want to use GPU acceleration, you can `pip install faiss-gpu`."
                " Otherwise, install faiss-cpu. For more, see https://github.com/facebookresearch/faiss"
            )

        self._init_embedding_system()

        logger.info(f"Loading FAISS index of {faiss_f}")
        self.faiss_index = faiss.read_index(faiss_f)

        # Requires faiss-gpu
        if use_cuda:
            try:
                self.faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.faiss_index)
            except AttributeError:
                raise AttributeError(
                    "You used --use_cuda but your FAISS installation does not support GPU."
                    " Uninstall faiss-cpu and install faiss-gpu or do not use --use_cuda."
                )

        self.faiss_index_size: int = self.faiss_index.ntotal
        logger.info(f"FAISS index loaded, containing {self.faiss_index_size:,} items")

        self.find_n_matches: Optional[int] = None
        self.min_score: float = 0.0

    def _init_embedding_system(self):
        """Initialises the embedding system, which can either be sent2vec or sentence transformers
        and also get the final output dimension of the representations."""
        if self.mode == "sent2vec":
            if not SENT2VEC_AVAILABLE:
                raise ImportError(
                    "Requested 'sent2vec', but module not installed. Install the right version from"
                    " https://github.com/epfml/sent2vec"
                )
            try:
                self.model = sent2vec.Sent2vecModel()
            except AttributeError as exc:
                raise AttributeError(
                    "'sent2vec' does not have attribute Sent2vecModel. You may have uninstalled an"
                    " incorrect version of sent2vec. The correct version can be found here:"
                    " https://github.com/epfml/sent2vec"
                ) from exc
            logger.info(f"Loading sent2vec model of {self.model_name_or_path}")
            self.model.load_model(self.model_name_or_path, inference_mode=True)
            self.hidden_size = self.model.get_emb_size()
        elif self.mode == "stransformers":
            if not STRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Requested 'stransformers', but module not installed. Please install the library"
                    " before continuing. https://github.com/UKPLab/sentence-transformers#installation"
                )
            logger.info(f"Loading SentenceTransformer model {self.model_name_or_path}")
            self.model = SentenceTransformer(self.model_name_or_path, device="cuda" if self.use_cuda else "cpu")
            self.hidden_size = self.model.encode(["This is a test ."]).shape[1]
        else:
            raise ValueError("'mode' must be 'sent2vec' or 'stransformers'")

    def _search(self, src_sent: str, sent_vec, query_multiplier: Optional[int] = None) -> List[tuple]:
        """Given a source sentence and its vector representation, query the FAISS index for similar
           sentences (cosine similarity). Query the index until `self.find_n_matches` are found.
           Matches cannot be identical to `src_sent`, no duplicate translations (from TM target) can be included,
           and the matching score must be at least `self.min_score`.
        :param src_sent: sentence to find the matches for
        :param query_multiplier: look for `query_multiplier * find_n_matches` matches to ensure that we
               find enough hits after filtering
        :return: a list of tuples, where each tuple contains: src_sent, the index of the match in the TM, TM src,
                 TM tgt, distance
        """
        # If the multiplier is set and if we are only looking for a subset of results,
        # look for find_n_matches * query_multiplier
        # Else: look for ALL results (=self.faiss_index_size)
        if query_multiplier and self.find_n_matches:
            # do not look for MORE than the actual size of the index
            search_topk = min(self.find_n_matches * query_multiplier, self.faiss_index_size)
        else:
            search_topk = self.faiss_index_size

        if self.use_cuda:
            # FAISS on GPU only queries supports up to max 1024
            search_topk = min(1024, search_topk)

        dists, idxs = self.faiss_index.search(sent_vec, search_topk)

        dists = dists.squeeze()
        idxs = idxs.squeeze()

        # exclude the matches that are an exact match
        # due to rounding errors dists can also be 0.999998 or 1.000004
        # so we also manually have to check whether remainding matches
        # are not exactly the same string-wise
        same = np.where(dists >= 1.0)
        real_dists = np.delete(dists, same)
        real_idxs = np.delete(idxs, same)

        # Now that have all results from our query we can filter these matches so that:
        #   - fuzzy_src is not equal to src_sent
        #   - fuzzy_tgt is not already part of a found result
        results = []
        # keep track of unique fuzzy targets. We do not want duplicates
        uniq_tgts = set()
        # we can abort when n_matches == find_n_matches so track n_matches
        # if find_n_matches is None we find all results
        n_matches = 0

        for i in range(real_dists.shape[0]):
            match_idx = real_idxs[i]
            # The corpora can be a list of sentences or a filename
            if isinstance(self.corpus_src, list):
                fuzzy_src = self.corpus_src[match_idx].rstrip()
            else:
                # linecache starts at 1 rather than 0
                fuzzy_src = linecache.getline(self.corpus_src, match_idx + 1).rstrip()

            # fuzzy match cannot be identical to input sentence
            if fuzzy_src == src_sent:
                continue

            if isinstance(self.corpus_tgt, list):
                fuzzy_tgt = self.corpus_tgt[match_idx].rstrip()
            else:
                fuzzy_tgt = linecache.getline(self.corpus_tgt, match_idx + 1).rstrip()

            if fuzzy_tgt in uniq_tgts:
                continue

            uniq_tgts.add(fuzzy_tgt)

            dist = real_dists[i]
            if self.min_score <= dist:
                results.append((src_sent, match_idx, fuzzy_src, fuzzy_tgt, real_dists[i]))
            n_matches += 1
            if self.find_n_matches is not None and n_matches == self.find_n_matches:
                break

        return results

    def search(
        self, src_sent: str, find_n_matches: Optional[int] = None, min_score: float = 0.0, query_multiplier: int = 2
    ) -> List[tuple]:
        """Converts a given `src_sent` into a vector with sent2vec. Then we find most similar matches in the TM
           with FAISS (in _search).
        :param src_sent: sentence to find the matches for
        :param find_n_matches: number of matches to find
        :param min_score: minimal matching score that a match must have
        :param query_multiplier: initially look for `query_multiplier * find_n_matches` matches to ensure that we
               find enough hits after filtering. If still not enough matches, search the whole index
        :return: a list of tuples, where each tuple contains: src_sent, the index of the match in the TM, TM src,
                 TM tgt, distance
        """
        self.find_n_matches = find_n_matches
        self.min_score = min_score

        if self.mode == "sent2vec":
            sent_vec = self.model.embed_sentence(src_sent)
        else:
            sent_vec = self.model.encode([src_sent], show_progress_bar=False)
        faiss.normalize_L2(sent_vec)

        # First, try to look for only double the requested matches. E.g. if we need 40, look for 80.
        # This increases the chance of finding all 40 matches after filtering, but we do not need
        # to search the WHOLE index. This saves a lot of time.
        results = self._search(src_sent, sent_vec, query_multiplier=query_multiplier)

        # If we did not find `find_n_matches` after filtering (because too many were filtered out),
        # do a brute-force search over the whole index. If the index does not contain too many duplicates,
        # this should only rarely happen
        # if find_n_matches is None, we already looked for all matches in the previous step
        if find_n_matches is not None and len(results) < find_n_matches:
            results = self._search(src_sent, sent_vec, query_multiplier=None)

        n_final_res = len(results)
        if find_n_matches is not None and n_final_res < find_n_matches:
            logger.debug(
                f"Only found {n_final_res:,} results (find_n_matches={find_n_matches:,}, min_score={min_score})"
                f" for source: {src_sent}"
            )

        return results
