"""Prepares your data for training an MT system. The script creates combinations of
   source and (possibly multiple) fuzzy target sentences, based on the initially created matches
   (cf. extraxt-fuzzy-matches). The current script can also filter features that need to be
   retained in the final files. Corresponding translations are also saved as well as those
   entries for which no matches were found."""

import logging
from math import floor
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm


logging.getLogger().setLevel(logging.INFO)


class DataAugmenter:
    def __init__(
        self,
        minscore: float,
        n_matches: int,
        combine: Optional[List[str]],
        is_trainset: bool = False,
        src_feats: Optional[List[str]] = None,
        fuzzy_tgt_feats: Optional[List[str]] = None,
        out_ranges: bool = False,
    ):
        if minscore >= 1 or minscore < 0:
            raise ValueError("Minscore should be a value between 0 and 1.")

        self.sf = src_feats if src_feats else []
        self.sf_sym = []
        if self.sf:
            if not set(self.sf).issubset({"matched", "side"}):
                raise ValueError("Source features must be a subset of None, 'matched', 'side'")

            if "side" in self.sf:
                self.sf_sym.append("S")
            if "matched" in self.sf:
                self.sf_sym.extend(["nm", "m"])

        self.ftf = fuzzy_tgt_feats if fuzzy_tgt_feats else []
        self.ftf_sym = []
        if fuzzy_tgt_feats:
            if not set(self.ftf).issubset({"matched", "side"}):
                raise ValueError("Fuzzy target features must be a subset of None, 'matched', 'side'")

            if "side" in self.ftf:
                self.ftf_sym.append("T")
            if "matched" in self.ftf:
                self.ftf_sym.extend(["nm", "m"])

        if len(self.sf) != len(self.ftf):
            raise ValueError("The number of source and fuzzy target features must be the same.")

        self.n_feats = len(self.sf)

        # Separator between source and fuzzy TGTs. Must have the same number of feats as the rest (here "￨B")
        self.sep = "@@@" + ("￨B" * self.n_feats)

        if combine not in {"nbest", "max_coverage"}:
            raise ValueError("Combination method must be one of 'nbest', 'max_coverage'")
        self.minscore = minscore
        self.is_trainset = is_trainset
        self.n_matches = n_matches
        self.combine = combine

        self.out_ranges = out_ranges

        self.src_lines = None
        self.tgt_lines = None
        self.fm_lines = None

    def filter_features(self, line: str, side: str) -> str:
        """Given a line, filter the token features to only match the requested features. So if the input line
        contains ￨S￨m but the requested feature is only "side", then ￨m will be removed. If no features are requested
        all features will be removed.

        :param line: an input line
        :param side: the side that this line belongs to ("src" or "tgt")
        :return: the line that only contains the requested features, possibly no features at all
        """
        allowed_feats = self.sf_sym if side == "src" else self.ftf_sym

        filtered_line = []
        for t in line.split():
            # If this token is the separator, just re-add the separator
            if t == self.sep:
                filtered_line.append(self.sep)
            else:
                word, *feats = t.split("￨")
                # If no features are needed, only include the word
                if not allowed_feats:
                    filtered_line.append(word)
                else:
                    feats = [f for f in feats if f in allowed_feats]

                    if len(feats) != self.n_feats:
                        raise ValueError(
                            f"Seems that you requested some features ({', '.join(self.sf)}) that are not"
                            " present in your input data. Please correct your command so that you only"
                            " request features that are already present in the input data. This"
                            " particular script can remove features, but not add new ones."
                        )

                    filtered_line.append("￨".join([word, *feats]))

        return " ".join(filtered_line)

    def process(self, src, tgt, fm, outdir):
        outfullsrc, outfulltgt, outnomatch, outmatch_ranges = self.create_outfiles(src, outdir)
        logging.info("Reading files ...")
        self.src_lines = Path(src).read_text().splitlines()
        self.tgt_lines = Path(tgt).read_text().splitlines()
        self.fm_lines = Path(fm).read_text().splitlines()

        """ Get n-best fuzzy matches from fuzzymatch file and store them in a dictionary """
        logging.info("Selecting and extracting fuzzy matches into a dictionary ...")
        fuzzy_dict = self.fuzzy2dict()

        """ For progressbar """
        logging.info("Augmenting source data ...")
        for s, t in tqdm(zip(self.src_lines, self.tgt_lines), total=len(self.src_lines)):
            try:
                # list of tuples of source, fuzzy tgt, score
                fmatches: List[Tuple[str, str, float]] = fuzzy_dict[s]
                # Filter features for search or add required ones
                sfeat = self.add_src_features(self.filter_features(fmatches[0][0], "src"))

                # Write the original source-target pair to the outfile
                if self.is_trainset:
                    outfullsrc.write(self.add_src_features(s) + "\n")
                    outfulltgt.write(t + "\n")

                aug_src = sfeat
                for fm in fmatches:
                    ftfeat = self.filter_features(fm[1], "tgt")
                    # concatenate each fuzzy match target
                    aug_src += f" {self.sep} {ftfeat}"

                outfullsrc.write(aug_src + "\n")
                outfulltgt.write(t + "\n")

                # Create output files for fuzzy match ranges only for non-training set
                if not self.is_trainset and outmatch_ranges:
                    maxscore = fmatches[0][-1]
                    outrangesrc = self.get_outfile_range(maxscore, outmatch_ranges, "src")
                    outrangetgt = self.get_outfile_range(maxscore, outmatch_ranges, "tgt")
                    outrangesrc.write(aug_src + "\n")
                    outrangetgt.write(t + "\n")

            except KeyError:
                # if no fuzzy match can be found write them to 'nomatch' file (if the option is set to True)
                if outnomatch:
                    outnomatch.write(s + "\n")
                """ Also write these segments to the full data
                Add the features to the source part first """
                outfullsrc.write(self.add_src_features(s) + "\n")
                outfulltgt.write(t + "\n")

        self.close_fhs([outfullsrc, outfulltgt, outnomatch, *outmatch_ranges])

    @staticmethod
    def close_fhs(fhs):
        """Close all file handles in a list"""
        for fh in fhs:
            fh.close()

    def add_src_features(self, line: str) -> str:
        """Add source (S) and/or no match (nm) features to a given source line if required. If a line already contains
        features, the line will be returned as-is.
        :param line: source line (with or without features)
        :return: a line with features if required (either the original line or edited)
        """
        if not self.sf_sym or "￨" in line:
            return line

        new_tokens = []
        for t in line.strip().split():
            # Add S and/or nm to source
            # Add an extra pipe before the features
            t += "￨" + "￨".join([f for f in self.sf_sym if f != "m"])
            new_tokens.append(t)
        return " ".join(new_tokens)

    @staticmethod
    def get_outfile_range(value, outmatch_ranges, side):
        if value < 1:
            for item in outmatch_ranges:
                # item example:
                # [0.9, 0.99, 'tgt', filename]
                if item[0] <= value < item[1] + 0.01 and side == item[2]:
                    return item[3]

        # In setsim returns a value larger than 1 place these items in the file 0.90-0.99 match range """
        if value >= 1:
            for item in outmatch_ranges:
                if item[0] == 0.9:
                    return item[3]

        # If none of the above works...
        raise ValueError(f"No files could be found for {value}")

    def create_outfiles(self, src, outdir):
        pdout = Path(outdir)

        out_matchranges = list()
        base_str = f"{Path(src).name}.min{self.minscore}.matches{self.n_matches}"
        outfullsrc = (pdout / f"{base_str}.ALL.source").open("w", encoding="utf-8")
        outfulltgt = (pdout / f"{base_str}.ALL.target").open("w", encoding="utf-8")

        """ Create output files per fuzzy match range only if out_ranges is True """
        if not self.out_ranges:
            outnomatch = None
            out_matchranges = None
        else:
            outnomatch = (pdout / f"{base_str}.NOMATCH.txt").open("w", encoding="utf-8")
            if not self.is_trainset:
                i = self.minscore
                while i < 1:
                    ri = round(i, 1)
                    if ri <= i:
                        ri = round(ri + 0.09, 2)
                    else:
                        ri = round(ri - 0.01, 2)
                    # Do not automatically format these lines...
                    # fmt: off
                    out_matchranges.append((i, ri, "src", (pdout / f"{base_str}.range{int(i * 100)}-{int(ri * 100)}.source").open("w", encoding="utf-8")))
                    out_matchranges.append((i, ri, "tgt", (pdout / f"{base_str}.range{int(i * 100)}-{int(ri * 100)}.target").open("w", encoding="utf-8")))
                    # fmt: on
                    i = ri + 0.01
        return outfullsrc, outfulltgt, outnomatch, out_matchranges

    @staticmethod
    def float_round(num, places=0, direction=floor):
        return direction(num * (10 ** places)) / float(10 ** places)

    def fuzzy2dict(self):
        pbar = tqdm(total=len(self.fm_lines))
        fdict = dict()
        for line in self.fm_lines:
            pbar.update()
            fields = line.strip().split("\t")
            tmsrc = fields[0]
            if "￨" in tmsrc:
                tmsrc_text = self.remove_features(tmsrc)
            else:
                tmsrc_text = tmsrc

            ftgt = fields[3]
            score = float(fields[4])
            # Skip the entry if the score is below the threshold
            if score < self.minscore:
                continue

            if tmsrc_text not in fdict:
                fdict[tmsrc_text] = [(tmsrc, ftgt, score)]
            else:
                """Check if ftgt already exists in the list of tuples
                If yes, do not save the current entry in the list"""
                ftgt_exists = False
                for item in fdict[tmsrc_text]:
                    ftgt_candidate = item[1]
                    if ftgt == ftgt_candidate:
                        ftgt_exists = True
                        break
                if not ftgt_exists:
                    fdict[tmsrc_text].append((tmsrc, ftgt, score))

        pbar.close()

        # Select the fuzzy matches for data augmentation
        sorted_fdict = dict()
        for key, value in fdict.items():
            sorted_values = sorted(value, key=lambda x: (x[-1]), reverse=True)

            if self.combine == "nbest":
                # Keep only n_matches best scores
                max_sorted_values = sorted_values[: self.n_matches]
                sorted_fdict[key] = max_sorted_values

            elif self.combine == "max_coverage":

                best_fm_src = sorted_values[0][0]
                len_src = self.get_str_length(best_fm_src)
                # Add token ids to avoid matching repeating tokens
                best_match_ids = [0]

                # Look for an additional match until we reach n
                for i in range(1, self.n_matches):
                    max_diff_tokens = 0
                    max_score = 0
                    best_match_id = -1
                    src_matching_union = self.get_matching_union(best_match_ids, sorted_values, len_src)
                    src_matching_union_ids = self.add_token_ids(src_matching_union)
                    match_found = False
                    for j in range(1, len(sorted_values)):
                        src = self.add_token_ids(sorted_values[j][0])
                        score = sorted_values[j][2]

                        # Get the list of non-overlapping tokens which are 'matches'
                        diff = list(set(src.split()) - set(src_matching_union_ids.split()))
                        diff_matched = self.remove_nonmatches(diff)

                        """ Consider this fuzzy match as the best candidate for combining with previous matches for max_coverage 
                        if the combination leads to more matching source tokens (diff_match) than the best candidate so far 
                        or it covers same amount of source tokens but its match score is higher (score) than previous best candidate """
                        if len(diff_matched) > max_diff_tokens or (
                            len(diff_matched) == max_diff_tokens and len(diff_matched) > 0 and score > max_score
                        ):

                            match_found = True
                            max_diff_tokens = len(diff_matched)
                            max_score = score
                            best_match_id = j

                    if match_found:
                        best_match_ids.append(best_match_id)
                    else:
                        """If no match is found that increases coverage, fall back to match with the best score:
                        Loop over all matches (sorted_values) and append the ID of the match with the highest match score
                        (but not already in best_match_ids) to best_match_ids"""
                        for k in range(1, len(sorted_values)):
                            if k in best_match_ids:
                                continue
                            else:
                                best_match_ids.append(k)
                                break
                sorted_fdict[key] = [sorted_values[ind] for ind in best_match_ids]

        return sorted_fdict

    @staticmethod
    def get_matching_union(list_ids, sorted_values, length):
        # Collect the tokens for each match ind
        tokens = [sorted_values[ind][0].strip().split() for ind in list_ids]

        union = []
        # Keep the matching source tokens in all 'tokens'
        for i in range(0, length):
            match_found = False
            for t in tokens:
                # Stop if no match/no-match feature is found in the first token of the source sentence
                if not (t[i].endswith("￨m") or t[i].endswith("￨nm")):
                    raise ValueError(
                        "Source text does not include match/no-match features: cannot perform max-coverage. You can either add the match/no-match features to source text or use 'nbest' instead."
                    )
                if t[i].endswith("￨m"):
                    union.append(t[i])
                    match_found = True
                    break
            """ If non-matching token is found, add the non-matching token from list of tokens 
                of the first fuzzy match source """
            if not match_found:
                union.append(tokens[0][i])
        return " ".join(union)

    @staticmethod
    def remove_features(line):
        text_tokens = []
        tokens = line.split()
        for t in tokens:
            features = t.split("￨")
            text_tokens.append(features[0])
        new_line = " ".join(text_tokens)
        return new_line

    @staticmethod
    def add_token_ids(line):
        tokens_ids = []
        tokens = line.strip().split()
        for i in range(0, len(tokens)):
            tokens_ids.append(str(i) + "￨" + tokens[i])
        new_line = " ".join(tokens_ids)
        return new_line

    @staticmethod
    def remove_nonmatches(token_list):
        new_token_list = []
        for t in token_list:
            if t.endswith("￨m"):
                new_token_list.append(t)
        return new_token_list

    @staticmethod
    def count_matching_tokens(line):
        tokens = line.strip().split()
        matching_tokens = [x for x in tokens if x.endswith("￨m")]
        return len(matching_tokens)

    @staticmethod
    def get_str_length(line):
        tokens = line.strip().split()
        return len(tokens)


def main():
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__)
    cparser.add_argument("--src", help="Input source file", required=True)
    cparser.add_argument("--tgt", help="Input target file", required=True)
    cparser.add_argument("--fm", help="File containing fuzzy matches for the input source", required=True)
    cparser.add_argument("--outdir", help="Output directory", required=True)
    cparser.add_argument("--minscore", help="Min. fuzzy match score threshold", required=True, type=float)
    cparser.add_argument(
        "--n_matches", help="Number of fuzzy target to be used in augmented source", required=True, type=int
    )
    cparser.add_argument(
        "--combine", help="Method of combining fuzzy matches", choices=["nbest", "max_coverage"], required=True
    )
    cparser.add_argument(
        "--is_trainset", help="Whether the input file the training set for the MT system", action="store_true"
    )
    cparser.add_argument(
        "--out_ranges",
        help="Whether to save augmented data for different fuzzy match range categories (considering the best fuzzy match score)",
        action="store_true",
    )
    cparser.add_argument(
        "-sf",
        "--src_feats",
        choices=["side", "matched"],
        nargs="+",
        help="Features to retain in the source tokens",
    )
    cparser.add_argument(
        "-ftf",
        "--fuzzy_tgt_feats",
        choices=["side", "matched"],
        nargs="+",
        help="Features to retain in the fuzzy target tokens",
    )

    cargs = cparser.parse_args()

    augmenter = DataAugmenter(
        cargs.minscore,
        cargs.n_matches,
        cargs.combine,
        cargs.is_trainset,
        cargs.src_feats,
        cargs.fuzzy_tgt_feats,
        cargs.out_ranges,
    )
    augmenter.process(
        cargs.src,
        cargs.tgt,
        cargs.fm,
        cargs.outdir,
    )


if __name__ == "__main__":
    main()
