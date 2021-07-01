"""Given source and target TM files, extract fuzzy matches for a new input file by using a
   variety of methods. You can use formal matching methods such as edit distance and set
   similarity, as well as semantic fuzzy matching with sent2vec and Sentence Transformers."""

import logging
import multiprocessing
import time
from multiprocessing.context import Process
from operator import itemgetter
from pathlib import Path
from typing import List, Optional, Tuple

import editdistance
import SetSimilaritySearch
from nfr.fuzzy_matching.faiss_retriever import FaissRetriever
from tqdm import tqdm


logger = logging.getLogger("nfr")


class FuzzyMatcher:
    def __init__(
        self,
        method: str,
        maxmatch: int,
        minscore: float,
        n_setsim_candidates: int,
        setsim_function: str,
        threads: int,
        model_name_or_path: Optional[str] = None,
        faiss: Optional[str] = None,
        use_cuda: bool = False,
        query_multiplier: int = 2,
    ):

        if method not in ["setsim", "setsimeditdist", "editdist", "sent2vec", "stransformers"]:
            raise ValueError(
                "Method should be one of the following: 'setsim', 'setsimeditdist', 'editdist', "
                "'sent2vec', 'stransformers'"
            )

        if method in ["sent2vec", "stransformers"] and not (model_name_or_path and faiss):
            raise ValueError(
                "When using method 'sent2vec' or 'stransformers', the 'model_name_or_path' and 'faiss'"
                " parameters must be provided"
            )

        self.match_count = 0
        self.nomatch_count = 0
        self.method = method
        self.insrc_lines = []
        self.tmsrc_lines = []
        self.tmtgt_lines = []
        self.maxmatch = maxmatch
        self.minscore = minscore
        self.n_setsim_candidates = n_setsim_candidates
        self.setsim_function = setsim_function
        self.n_threads = threads
        self.index = None

        self.model_name_or_path = model_name_or_path
        self.faiss_f = faiss
        self.faiss_retriever = None
        self.use_cuda = use_cuda
        self.query_multiplier = query_multiplier

        self.results_q = None

        if self.use_cuda and self.n_threads > 1:
            raise ValueError(
                "Cannot use 'use_cuda' alongside multithreading ('n_threads' > 1). Either use 'use_cuda',"
                " or 'n_threads' but not at the same."
            )

    def _get_unique_chunks(self, input_lines) -> List:
        """Split a list of unique items into N equal parts
        :param input_lines: list of unique items
        :return: list of lists (of size "n_threads")
        """
        unique_lines = list(set(input_lines))
        length = len(unique_lines)
        logger.info("No. unique segments in 'insrc' = " + str(length))
        return [
            unique_lines[i * length // self.n_threads : (i + 1) * length // self.n_threads]
            for i in range(self.n_threads)
        ]

    def _init_setsim_index(self, tmsrc_lines):
        """
        Initialize SetSimilarity Search index
        """
        if self.method in ["setsimeditdist", "setsim"]:
            # Initialize setsim search index using TM source
            segset = []
            for line in tmsrc_lines:
                tokens = line.strip().split()
                segset.append(tokens)
            index = SetSimilaritySearch.SearchIndex(
                segset, similarity_func_name=self.setsim_function, similarity_threshold=self.minscore
            )
            self.index = index

    @staticmethod
    def _tuple2string(tup: Tuple) -> str:
        """
        Convert a tuple of tokens to string
        :param tup: an existing tuple
        :return: string
        """

        new_tup = tuple(
            str(x).replace("\t", " ") for x in tup
        )  # replace all tabs with spaces before using the tab as delimiter
        new_tup_str = "\t".join(new_tup)
        return new_tup_str

    @staticmethod
    def _readlines(fin):
        with open(fin, encoding="utf-8") as fhin:
            lines = fhin.readlines()
        return lines

    @staticmethod
    def _remove_duplicate_matches(matches):
        # format of matches_list: (source, id, candidate, tmtgt_lines[id].strip(), final_score)
        seen_translations = set()
        matches_unique_translations = list()
        for item in matches:
            # Translation is stored at item with index -2
            translation = item[-2]
            if translation in seen_translations:
                continue
            else:
                matches_unique_translations.append(item)
                seen_translations.add(translation)
        return matches_unique_translations

    @staticmethod
    def _get_editdistance(source: str, candidate: str) -> float:
        """
        Get editdistance score between two lists of word tokens
        :param source: list of tokens for the input sentence
        :param candidate: list of tokens for the candidate sentence
        :return: return editdistance score (normalized on sentence length)
        """
        candidate = candidate.split()
        source = source.split()

        ed = editdistance.eval(source, candidate)
        maxlength = max(len(source), len(candidate))
        ed_norm = (maxlength - ed) / maxlength

        return ed_norm

    def _init_data(self, insrc, tmsrc, tmtgt):
        """Initialize instance attributes based on the input that `process` received"""
        self.tmsrc_lines = self._readlines(tmsrc)
        self.tmtgt_lines = self._readlines(tmtgt)

        if len(self.tmsrc_lines) != len(self.tmtgt_lines):
            raise ValueError("No. lines in tmsrc and tmtgt are not equal.")

        self.insrc_lines = self._readlines(insrc)

    def _init_index(self):
        if self.method in ["sent2vec", "stransformers"] and self.faiss_retriever is None:
            self.faiss_retriever = FaissRetriever(
                self.tmsrc_lines, self.tmtgt_lines, self.model_name_or_path, self.faiss_f, self.method, self.use_cuda
            )
        elif self.index is None:
            self._init_setsim_index(self.tmsrc_lines)

    def process(self, insrc, tmsrc, tmtgt):
        start_time = time.time()
        self._init_data(insrc, tmsrc, tmtgt)
        self._init_index()

        fout = f"{insrc}.matches.mins{self.minscore}.maxm{self.maxmatch}"
        fout += (
            f".{self.method}"
            if self.method in ["sent2vec", "stransformers"]
            else f".{self.setsim_function}{self.n_setsim_candidates}"
        )
        fout += ".txt"

        with multiprocessing.Manager() as manager:
            # Start queue where we'll `put` the results so that the writer can `get` them
            self.results_q = manager.Queue()
            # Separate writer process for efficiency reasons
            # (Might not matter _that_ much depending on your chosen batch size)
            writer_proc = Process(target=self._writer, args=(fout,))
            writer_proc.start()

            # If we only use 0/1 thread, just run in the main thread. This will ensure that we do not run into issues
            # with FAISS on GPU when using use_cuda
            if self.n_threads < 2:
                self._match(self.insrc_lines, 0)
            else:
                # Get the unique source sentences in insrc and split the data into chunks for multithreading
                unique_insrc_chunks = self._get_unique_chunks(self.insrc_lines)
                arg_list = []
                for i in range(self.n_threads):
                    arg_list.append((unique_insrc_chunks[i], i))

                processes = []
                for i in range(self.n_threads):
                    p = multiprocessing.Process(target=self._match, args=(arg_list[i]))
                    processes.append(p)
                    p.start()

                for process in processes:
                    process.join()

            self.results_q.put("done")
            writer_proc.join()
            writer_proc.terminate()

        logger.info("Extracting fuzzy matches took " + str(time.time() - start_time) + " to run")

    def _writer(self, fout):
        """The writer process that writes the output as the expected format.
        Intended to be run in a separate process that reads input from a queue and writes
        it to an output file."""
        with Path(fout).open("w", encoding="utf-8") as fhout:
            while True:
                # Fetch items from the queue
                m = self.results_q.get()
                # `break` if the item is 'done' (put there in `self.process()`)
                if m == "done":
                    break

                for tup in m:
                    tup_str = self._tuple2string(tup)
                    fhout.write(tup_str + "\n")

                fhout.flush()

        logger.info(f"Output written to {fout}")

    def _match(self, input_lines, thread_id):
        # Only show progress bar for the first process.
        for i in tqdm(range(len(input_lines)), disable=thread_id != 0, desc="Progress process #0"):
            matches = []
            source = input_lines[i].strip()
            source_tok = source.split()

            if self.method in ["sent2vec", "stransformers"]:
                matches = self.faiss_retriever.search(source, self.maxmatch, self.minscore, self.query_multiplier)
            else:
                if self.index is not None:
                    # Query the setsim index to collect high fuzzy match candidates
                    result = self.index.query(source_tok)
                    # Query result the format [(matchindex, similarity score)]
                    # Sort the results on similarity score
                    result.sort(key=itemgetter(1), reverse=True)
                    # Take the most similar n matches
                    result = result[: self.n_setsim_candidates]

                    # Get the similarity score for each candidate
                    for r in result:
                        idx = r[0]
                        # Keep the original string to write
                        candidate = self.tmsrc_lines[idx].strip()
                        # Skip if source and candidate are the same
                        if source == candidate:
                            continue

                        if self.method == "setsim":
                            # Keep setsim score
                            final_score = r[1]
                        elif self.method == "setsimeditdist":
                            # Calculate editdistance
                            final_score = self._get_editdistance(source, candidate)
                        else:
                            pass

                        # keep the match if within the threshold
                        if self.minscore <= final_score:
                            matches.append((source, idx, candidate, self.tmtgt_lines[idx].strip(), final_score))

                # Get matches using editdistance only
                elif self.method == "editdist":
                    for j in range(len(self.tmsrc_lines)):
                        candidate = self.tmsrc_lines[j].strip()
                        # Skip if source and candidate are the same
                        if source == candidate:
                            continue

                        ed_norm = self._get_editdistance(source, candidate)
                        if self.minscore <= ed_norm:
                            matches.append((source, j, candidate, self.tmtgt_lines[j].strip(), ed_norm))

                if matches:
                    self.match_count += 1
                    # Keep matches only with unique translations (keep only one element with the same translation)
                    matches = self._remove_duplicate_matches(matches)
                    # Sort the matches based on match score and keep the best matches (maxmatch)
                    sorted_matches = sorted(matches, key=lambda x: (x[-1]), reverse=True)
                    matches = sorted_matches[: self.maxmatch]

                else:
                    self.nomatch_count += 1

            self.results_q.put(matches)

        return


def main():
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__)
    cparser.add_argument(
        "--tmsrc", help="Source text of the TM from which fuzzy matches will be extracted", required=True
    )
    cparser.add_argument(
        "--tmtgt", help="Target text of the TM from which fuzzy matches will be extracted", required=True
    )
    cparser.add_argument(
        "--insrc", help="Input source file to extract matches for (insrc is queried against tmsrc)", required=True
    )
    cparser.add_argument(
        "--method",
        help="Method to find fuzzy matches",
        choices=["editdist", "setsim", "setsimeditdist", "sent2vec", "stransformers"],
        required=True,
    )

    cparser.add_argument(
        "--minscore",
        help="Min fuzzy match score. Only matches with a" " similarity score of at least 'minscore' will be included",
        required=True,
        type=float,
    )
    cparser.add_argument(
        "--maxmatch", help="Max number of fuzzy matches kept per source segment", required=True, type=int
    )
    cparser.add_argument(
        "--model_name_or_path",
        help="Path to sent2vec model (when `method` is sent2vec) or sentence-transformers model name"
        " when method is stransformers (see https://www.sbert.net/docs/pretrained_models.html)",
    )
    cparser.add_argument(
        "--faiss", help="Path to faiss index. Must be provided when `method` is sent2vec or stransformers"
    )
    cparser.add_argument(
        "--threads", help="Number of threads. Must be 0 or 1 when using `use_cuda`", default=1, type=int
    )

    cparser.add_argument(
        "--n_setsim_candidates", help="Number of fuzzy match candidates extracted by setsim", type=int, default=2000
    )
    cparser.add_argument(
        "--setsim_function", help="Similarity function used by setsimsearch", type=str, default="containment_min"
    )
    cparser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Whether to use GPU for FAISS indexing and sentence-transformers. For this to work"
        " properly `threads` should be 0 or 1.",
    )
    cparser.add_argument(
        "-q",
        "--query_multiplier",
        help="(applies only to FAISS) Initially look for `query_multiplier * maxmatch`"
        " matches to ensure that we find enough hits after filtering. If still not"
        " enough matches, search the whole index",
        type=int,
        default=2,
    )
    cparser.add_argument(
        "-v",
        "--logging_level",
        choices=["info", "debug"],
        help="Set the information level of the logger. 'info' shows trivial information about the process. 'debug'"
        " also notifies you when less matches are found than requested during semantic matching ",
        default="info",
    )

    cargs = cparser.parse_args()

    logger.setLevel(cargs.logging_level.upper())

    matcher = FuzzyMatcher(
        cargs.method,
        cargs.maxmatch,
        cargs.minscore,
        cargs.n_setsim_candidates,
        cargs.setsim_function,
        cargs.threads,
        cargs.model_name_or_path,
        cargs.faiss,
        cargs.use_cuda,
        cargs.query_multiplier,
    )
    matcher.process(cargs.insrc, cargs.tmsrc, cargs.tmtgt)


if __name__ == "__main__":
    main()
