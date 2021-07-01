"""Given a file containing source, fuzzy source and fuzzy target columns, finds the
   tokens in fuzzy_src that match with src according to the edit distance metric. Then
   the indices of those matches are used together with the word alignments (GIZA) between
   fuzzy_src and fuzzy_tgt to mark fuzzy target tokens with ￨m (match) or ￨nm (no match).
   This feature indicates whether or not the fuzzy_src token that is aligned with said fuzzy
   target token has a match in the original source sentence. The feature is also added to source
   tokens when a match was found according to the methodology described above.
   In addition, a "side" feature is added. This indicates which side the token is from,
   ￨S (source) or ￨T (target).

   So, in sum, every source and fuzzy target token will have two features: match/no-match and
   its side. These features can be filtered in the next processing step, nfr-augment-data."""

import linecache
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from nfr.utils import get_n_lines
from nltk import edit_distance, edit_distance_align
from tqdm import tqdm


class FeaturesForTraining:
    def __init__(
        self,
        fin: Union[str, bytes, PathLike],
        falign: Union[str, bytes, PathLike],
        out: Optional[Union[str, bytes, PathLike]] = None,
        verbose: bool = False,
    ):
        self.pfin = Path(fin).resolve()

        # Linecache.readline does not throw errors but returns '' instead. So check whether file exists beforehand
        if not Path(falign).exists() or not Path(falign).is_file():
            raise ValueError("'falign' not found or not a file.")

        # Linecache requires string rather than path
        self.falign = str(Path(falign).resolve())
        self.pfout = out if out else self.pfin.with_suffix(".trainfeats" + self.pfin.suffix)
        self.verbose = verbose

    def add_fuzzy_matched_feature(self, row: pd.Series, align_str: str) -> Tuple[List[str], List[str]]:
        """Adds a feature to fuzzy match target tokens to indicate whether its aligned
           fuzzy_src equivalent has a match in the original source.
        :param row: pandas row
        :param align_str: alignment string representing the alignment between fuzzy source and fuzzy target
        :return: the target sentence as a string where the features have been added
        """
        if not align_str:
            raise KeyError(
                f"The given alignments for index {row.iloc[1]} are empty or not available. This means that"
                " either this line in the alignment file is empty, or that the alignment file does not have"
                f" as many lines as this index ({row.iloc[1]} ), or that the file is currently not"
                " accessible.\nThis can indicate that the alignments file is in use by another process, or"
                " that you did not use the full train file to create the word alignments."
            )

        # alignments between fuzzy source and fuzzy target
        # src_idx: [tgt_idx, tgt_idx, ...]
        fuzzy_src2tgt_aligns: Dict[int, List[int]] = self.aligns_from_str_to_dict(align_str)

        src: List[str] = row.iloc[0].split()
        fuzzy_src: List[str] = row.iloc[2].split()
        fuzzy_tgt: List[str] = row.iloc[3].split()

        if self.verbose:
            row_str = f"ROW_IDX {int(row.iloc[1]):,}\n"
            row_str += "=" * len(row_str)
            print(row_str)
            print("SRC", src)
            print("FUZZY_SRC", fuzzy_src)
            print("FUZZY_TGT", fuzzy_tgt)
            print()

        # Find the alignments of identical tokens between source and fuzzy source
        # Target side is not a list because we are only interested in the first occurrence of a token
        # in such rare cases where one source token is aligned with two identical target tokens
        # src_idx: fuzzy_src_idx
        aligned_matches: Dict[int, int] = self.get_matches(src, fuzzy_src)

        # Collect all fuzzy target indices that are aligned to fuzzy source items that were aligned/matched with source
        all_fuzzy_tgt_matches: Set[int] = set()
        for matched_fuzzy_src_idx in aligned_matches.values():
            # In rare cases a matched item (between src and fuzzy_src)
            # is not aligned between fuzzy_src and fuzzy_tgt, in which case we ignore it
            if matched_fuzzy_src_idx in fuzzy_src2tgt_aligns:
                all_fuzzy_tgt_matches.update(fuzzy_src2tgt_aligns[matched_fuzzy_src_idx])

        fuzzy_tgt_feat: List[str] = ["m" if idx in all_fuzzy_tgt_matches else "nm" for idx in range(len(fuzzy_tgt))]
        src_feat: List[str] = ["m" if idx in aligned_matches.keys() else "nm" for idx in range(len(src))]

        if self.verbose:
            print("MATCHED + ALIGNED\n-----------------")
            print("ALIGNED FUZZY MATCH TGT IDXS", all_fuzzy_tgt_matches)
            print("SRC_FEAT", " ".join(src_feat))
            print("FUZZY_TGT_FEAT", " ".join(fuzzy_tgt_feat))
            print("\n")

        return src_feat, fuzzy_tgt_feat

    @staticmethod
    def aligns_from_str_to_dict(align_str: str) -> Dict[int, List[int]]:
        """Converts a GIZA string into a dictionary of src_idxs: [tgt_idx, tgt_idx, ...]
        :param align_str: input GIZA-like string
        :return: dictionary containg source-to-target alignments where the target side is a list
        """
        src2tgt = defaultdict(list)
        for pair in align_str.split(" "):
            src_idx, tgt_idx = map(int, pair.split("-"))
            src2tgt[src_idx].append(tgt_idx)
        return dict(src2tgt)

    def get_matches(self, src: List[str], fuzzy_src: List[str]):
        """Find the alignments between source and fuzzy source where tokens are identical. By "alignments",
           we mean the alignments according to NLTK's optimal path that is used to calculate edit distance.
           So we are interested in finding those tokens where the edit distance algorithm thinks two tokens
           are "matched".
        :param src: the original source tokens
        :param fuzzy_src: the fuzzy source tokens
        :return: a dictionary of src_idx: tgt_idx, indicating the index of a source token that is matched with a
                 target token. If a source token happens to be matched with multiple target tokens, we only return
                 the first matched target token.
        """
        alignments: List[Tuple[int, int]] = edit_distance_align(src, fuzzy_src)

        # (0, 0) is the "starting point" in the alignment matrix. Indices start at 1
        # To be compatible with GIZA output, we subtract by 1 so that indices are zero-based
        alignments = [(src_idx - 1, tgt_idx - 1) for src_idx, tgt_idx in alignments if src_idx > 0 and tgt_idx > 0]

        aligned_matches: Dict[int, int] = {}
        for src_idx, tgt_idx in alignments:
            # By checking if the src_idx already exists in the dict, we ensure that one source
            # item can only match with one target item (the first one in the target sentence)
            if src[src_idx] == fuzzy_src[tgt_idx] and src_idx not in aligned_matches:
                aligned_matches[src_idx] = tgt_idx

        if self.verbose:
            print("MATCHES BETWEEN SRC AND FUZZY_SRC\n---------------------------------")
            print("EDIT DIST. ALIGNMENTS", alignments)
            print("EDIT DISTANCE", edit_distance(src, fuzzy_src))
            print(
                "MATCHED TOKENS", [(src[src_idx], fuzzy_src[tgt_idx]) for src_idx, tgt_idx in aligned_matches.items()]
            )
            print(
                "ALIGNMENT OF MATCHED TOKENS", " ".join(["-".join(map(str, pair)) for pair in aligned_matches.items()])
            )
            print()

        return aligned_matches

    @staticmethod
    def join_features(*feats):
        """Joins multiple feature lists of strings and turns them into one list where features have
           been combined and separated by "￨" (unicode character FFE8).
        :param feats: feature lists
        :return: single list of joined features
        """
        lengths = set(map(len, feats))

        # len can be 0 if no features were requested, e.g. self.src_feats = None
        if len(lengths) > 1:
            raise ValueError("All feature lists must be of the same length")

        return ["￨".join(feat_tuple) for feat_tuple in zip(*feats)]

    def process_row(self, row):
        """Processes a single row. Adds different features together as requested. Required features
        are set in self.{src,tgt}_feats. None will not add any features, "matched" will add "m" for
        a matched token and "nm" for a token with no match, "side" will add "S" to source tokens and
        "T" to target tokens. Combinations are possible, e.g. ["side", "matched"] will lead to <src_tok>￨S￨m"""
        src_tokens = row.iloc[0].split()
        fuzzy_tgt_tokens = row.iloc[3].split()

        src_feats = []
        fuzzy_tgt_feats = []

        src_feats.append(["S"] * len(src_tokens))
        fuzzy_tgt_feats.append(["T"] * len(fuzzy_tgt_tokens))

        # The index in the fuzzy file (iloc[1]) are the linenumbers from the original training file
        # (zero-based - so do +1). So to get alignments that correspond to the correct index, we can just get
        # the line at that index - but linecache expects line numbers (1-indexed), so do +1
        # Note that we could also have used .readlines() to read all lines in memory, but to accommodate
        # larger-than-memory input files and prevent OOM, we use linecache
        # NOTE: linecache.getline does not throw errors. If self.align is locked, in use, or otherwise not
        # accessible .getline returns '' rather than throwing an error. The reason of the error is hard to catch.
        # Instead, we show a notice in the warning in .add_fuzzy_matched_feature that it may be that the
        # file is in use.
        src_m_feat, fuzzy_tgt_m_feat = self.add_fuzzy_matched_feature(
            row, linecache.getline(self.falign, int(row.iloc[1]) + 1).rstrip()
        )
        src_feats.append(src_m_feat)
        fuzzy_tgt_feats.append(fuzzy_tgt_m_feat)

        src_feats = self.join_features(*src_feats)
        fuzzy_tgt_feats = self.join_features(*fuzzy_tgt_feats)

        src_feat_str = " ".join(["￨".join(tupe) for tupe in zip(src_tokens, src_feats)])
        fuzzy_tgt_feat_str = " ".join(["￨".join(tupe) for tupe in zip(fuzzy_tgt_tokens, fuzzy_tgt_feats)])

        return src_feat_str, fuzzy_tgt_feat_str

    def add_feats(self):
        """In-memory entry point to find out whether a fuzzy target token's aligned fuzzy source token has a match in
        the original source sentence. Reads in `fin` as a dataframe to easily process the data with
        `DataFrame.apply`. Writes modified dataframe (fuzzy target is replaced by factored fuzzy target) to
        an output file.
        """
        df = pd.read_csv(self.pfin, sep="\t", header=None)

        # For progressbar
        tqdm.pandas()

        # Collect the results and re-assign columns
        df.iloc[:, 0], df.iloc[:, 3] = zip(*df.progress_apply(lambda row: self.process_row(row), axis=1))

        df.to_csv(self.pfout, sep="\t", index=False, header=False)

    def add_feats_lazy(self):
        """Lazy entry point to find out whether a fuzzy target token's aligned fuzzy source token has a match in the
        original source sentence. Reads in line-per-line from `fin`. Writes modified lines to an output file.
        """
        if self.verbose:
            print(f"Getting number of lines in file {self.pfin}...")
        n_lines = get_n_lines(self.pfin)

        with self.pfin.open(encoding="utf-8") as fhin, self.pfout.open("w", encoding="utf-8") as fhout:
            for line in tqdm(fhin, total=n_lines, unit="sent"):
                line = line.rstrip().split("\t")
                if not line:
                    continue

                line[0], line[3] = self.process_row(pd.Series(line))
                fhout.write("\t".join(line) + "\n")


def main():
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__)
    cparser.add_argument("fin", help="Input file")
    cparser.add_argument("falign", help="Alignment file")
    cparser.add_argument(
        "-o", "--out", help="Output file. If not given, will use the input file with '.trainfeats' before the suffix"
    )
    cparser.add_argument(
        "-l", "--lazy", action="store_true", help="Whether to use lazy processing. Useful for very large files"
    )
    cparser.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to print intermediate results to stdout"
    )

    cargs = vars(cparser.parse_args())
    do_lazy = cargs.pop("lazy")
    processor = FeaturesForTraining(**cargs)

    if do_lazy:
        processor.add_feats_lazy()
    else:
        processor.add_feats()


if __name__ == "__main__":
    main()
