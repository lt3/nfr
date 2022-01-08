# Neural fuzzy repair


## Installation
For basic usage you can simply install the library from pip

```bash
pip install nfr
```

... or clone from git and install.

```bash
git clone https://github.com/lt3/nfr.git
cd nfr
pip install .
```

By default, semantic matching capabilities with sent2vec and Sentence Transformers are not enabled because the
 dependencies are considerably large. If you want to enable semantic matching, you need to install FAISS and one of
 Sentence Transformers or Sent2Vec.

- [FAISS](https://github.com/facebookresearch/faiss) (`pip install faiss-cpu` or `pip install faiss-gpu`)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) (`pip install sentence-transformers`)
  - Sentence Transformers relies on PyTorch. Depending on your OS, it might be that a CPU-version of `torch` will be
     installed by default. If you want better performance, and you have a CUDA-enabled device avaialble, it is
     recommended to install a  CUDA-enabled version of [`torch`](https://pytorch.org/get-started/locally/) before 
     installing `sentence-transformers`.
- [Sent2Vec](https://github.com/epfml/sent2vec) (clone and install from GitHub; do *not* use pip as that is a 
  different version)     

## Usage

After installation, four commands are exposed. In all cases, you can type `<command> -h` for these usage instructions.

1. `nfr-create-faiss-index`: Creates a FAISS index for semantic matches with Sent2Vec or Sentence Transformers.
    This is a necessary step if you want to extract semantic fuzzy matches later on.
    
```
usage: nfr-create-faiss-index [-h] -c CORPUS_F -p MODEL_NAME_OR_PATH -o
                              OUTPUT_F [-m {sent2vec,stransformers}]
                              [-b BATCH_SIZE] [--use_cuda]

Create a FAISS index based on the semantic representation of an existing text
corpus. To do so, the text will be embedded by means of a sent2vec model or a
sentence-transformers model. The index is (basically) an efficient list that
contains all the representations of the training corpus sentences (the TM). as
such, this index can later be used to find those entries that are most similar
to a given representation of a sentence. The index is saved to a binary file
so that it can be reused later on to calculate cosine similarity scores and to
retrieve the most resembling entries.

optional arguments:
  -h, --help            show this help message and exit
  -c CORPUS_F, --corpus_f CORPUS_F
                        Path to the corpus to turn into vectors and add to the
                        index. This is typically your TM or training file for
                        an MT system containing text, one sentence per line
  -p MODEL_NAME_OR_PATH, --model_name_or_path MODEL_NAME_OR_PATH
                        Path to sent2vec model (when `method` is sent2vec) or
                        sentence-transformers model name when method is
                        stransformers (see
                        https://www.sbert.net/docs/pretrained_models.html)
  -o OUTPUT_F, --output_f OUTPUT_F
                        Path to the output file to write the FAISS index to
  -m {sent2vec,stransformers}, --mode {sent2vec,stransformers}
                        Whether to use 'sent2vec' or 'stransformers'
                        (sentence-transformers)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size to use to create sent2vec embeddings or
                        sentence-transformers embeddings. A larger value will
                        result in faster creation, but may lead to an out-of-
                        memory error. If you get such an error, lower the
                        value.
  --use_cuda            Whether to use GPU when using sentence-transformers.
                        Requires PyTorch installation with CUDA support and a
                        CUDA-enabled device
```
    
    
2. `nfr-extract-fuzzy-matches`: Here, fuzzy matches can be extracted from the training set. A variety of options are
    available, including semantic fuzzy matching, setsimilarity and edit distance.
    
```
usage: nfr-extract-fuzzy-matches [-h] --tmsrc TMSRC --tmtgt TMTGT --insrc
                                 INSRC --method
                                 {editdist,setsim,setsimeditdist,sent2vec,stransformers}
                                 --minscore MINSCORE --maxmatch MAXMATCH
                                 [--model_name_or_path MODEL_NAME_OR_PATH]
                                 [--faiss FAISS] [--threads THREADS]
                                 [--n_setsim_candidates N_SETSIM_CANDIDATES]
                                 [--setsim_function SETSIM_FUNCTION]
                                 [--use_cuda] [-q QUERY_MULTIPLIER]
                                 [-v {info,debug}]

Given source and target TM files, extract fuzzy matches for a new input file
by using a variety of methods. You can use formal matching methods such as
edit distance and set similarity, as well as semantic fuzzy matching with
sent2vec and Sentence Transformers.

optional arguments:
  -h, --help            show this help message and exit
  --tmsrc TMSRC         Source text of the TM from which fuzzy matches will be
                        extracted
  --tmtgt TMTGT         Target text of the TM from which fuzzy matches will be
                        extracted
  --insrc INSRC         Input source file to extract matches for (insrc is
                        queried against tmsrc)
  --method {editdist,setsim,setsimeditdist,sent2vec,stransformers}
                        Method to find fuzzy matches
  --minscore MINSCORE   Min fuzzy match score. Only matches with a similarity
                        score of at least 'minscore' will be included
  --maxmatch MAXMATCH   Max number of fuzzy matches kept per source segment
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to sent2vec model (when `method` is sent2vec) or
                        sentence-transformers model name when method is
                        stransformers (see
                        https://www.sbert.net/docs/pretrained_models.html)
  --faiss FAISS         Path to faiss index. Must be provided when `method` is
                        sent2vec or stransformers
  --threads THREADS     Number of threads. Must be 0 or 1 when using
                        `use_cuda`
  --n_setsim_candidates N_SETSIM_CANDIDATES
                        Number of fuzzy match candidates extracted by setsim
  --setsim_function SETSIM_FUNCTION
                        Similarity function used by setsimsearch
  --use_cuda            Whether to use GPU for FAISS indexing and sentence-
                        transformers. For this to work properly `threads`
                        should be 0 or 1.
  -q QUERY_MULTIPLIER, --query_multiplier QUERY_MULTIPLIER
                        (applies only to FAISS) Initially look for
                        `query_multiplier * maxmatch` matches to ensure that
                        we find enough hits after filtering. If still not
                        enough matches, search the whole index
  -v {info,debug}, --logging_level {info,debug}
                        Set the information level of the logger. 'info' shows
                        trivial information about the process. 'debug' also
                        notifies you when less matches are found than
                        requested during semantic matching
```

3. `nfr-add-training-features`: Adds features to the input. These involve the side of a token (source token or fuzzy
   target) or whether or not a token was matched.
   
```
usage: nfr-add-training-features [-h] [-o OUT] [-l] [-v] fin falign

Given a file containing source, fuzzy source and fuzzy target columns, finds
the tokens in fuzzy_src that match with src according to the edit distance
metric. Then the indices of those matches are used together with the word
alignments (GIZA) between fuzzy_src and fuzzy_tgt to mark fuzzy target tokens
with ￨m (match) or ￨nm (no match). This feature indicates whether or not the
fuzzy_src token that is aligned with said fuzzy target token has a match in
the original source sentence. The feature is also added to source tokens when
a match was found according to the methodology described above. In addition, a
"side" feature is added. This indicates which side the token is from, ￨S
(source) or ￨T (target). So, in sum, every source and fuzzy target token will
have two features: match/no-match and its side. These features can be filtered
in the next processing step, nfr-augment-data.

positional arguments:
  fin                Input file
  falign             Alignment file

optional arguments:
  -h, --help         show this help message and exit
  -o OUT, --out OUT  Output file. If not given, will use the input file with
                     '.trainfeats' before the suffix
  -l, --lazy         Whether to use lazy processing. Useful for very large
                     files
  -v, --verbose      Whether to print intermediate results to stdout
```

4. `nfr-augment-data`: Prepares the dataset to be used in an MT system. Allows you to combine fuzzy matches and choose
   features to use.

```
usage: nfr-augment-data [-h] --src SRC --tgt TGT --fm FM --outdir OUTDIR
                        --minscore MINSCORE --n_matches N_MATCHES --combine
                        {nbest,max_coverage} [--is_trainset] [--out_ranges]
                        [-sf {side,matched} [{side,matched} ...]]
                        [-ftf {side,matched} [{side,matched} ...]]

Prepares your data for training an MT system. The script creates combinations
of source and (possibly multiple) fuzzy target sentences, based on the
initially created matches (cf. extraxt-fuzzy-matches). The current script can
also filter features that need to be retained in the final files.
Corresponding translations are also saved as well as those entries for which
no matches were found.

optional arguments:
  -h, --help            show this help message and exit
  --src SRC             Input source file
  --tgt TGT             Input target file
  --fm FM               File containing fuzzy matches for the input source
  --outdir OUTDIR       Output directory
  --minscore MINSCORE   Min. fuzzy match score threshold
  --n_matches N_MATCHES
                        Number of fuzzy target to be used in augmented source
  --combine {nbest,max_coverage}
                        Method of combining fuzzy matches
  --is_trainset         Whether the input file the training set for the MT
                        system
  --out_ranges          Whether to save augmented data for different fuzzy
                        match range categories (considering the best fuzzy
                        match score)
  -sf {side,matched} [{side,matched} ...], --src_feats {side,matched} [{side,matched} ...]
                        Features to retain in the source tokens
  -ftf {side,matched} [{side,matched} ...], --fuzzy_tgt_feats {side,matched} [{side,matched} ...]
                        Features to retain in the fuzzy target tokens
```

## Best Configuration: Step-by-step guide 
The best configuration (for majoirty of the language pairs tested in Tezcan, Bulté & Vanroy; 2021) consists of the following parameters/properties: <br> 
- Preprocessing: Tokenization, Truecasing and sub-word segmentation using Byte-Pair Encoding (BPE) with a merged vocabulary of 32K of the source and target languages (these preprocessing steps are performed before Step 1 below). We used [Moses Toolkit](https://www.statmt.org/moses/) for tokenization and truecasing and OpenNMT for BPE, which relies on [the original BPE implementation](https://github.com/rsennrich/subword-nmt). <br> 
- Fuzzy matching using cosine similarity between segments using sent2vec (with a min. fuzzy match score of 0.5 and max. 40 fuzzy matches per source segment) <br> 
- Data augmentation using "maximum coverage" using 2 fuzzy matches per segment with features ("source" feature on the source tokens, "match/no-match" feature on fuzzy-match target tokens) <br> 

**Note**: You can follow along with this step-by-step guide by making use of a dummy data set that we provide on the
[release page](https://github.com/lt3/nfr/releases/tag/v1.0.0). This dataset and the supplementary models and indices
are purposefully kept small in size. Therefore, their performance is not great. They merely are intended to show you how
to use our code.

### Step 1. Extract Fuzzy Matches (preprocessed data)
Fuzzy matches need to be extracted for the training, test and development sets **separately**. <br>
Fuzzy matching using sent2vec requires a _sent2vec model_ built on the source side (source language) of the (preprocessed) training set and a _FAISS index_ for this model. Please see sent2vec documentation on how to build a sent2vec model and our paper for the parameters we used in our experiments.

To generate a FAISS index (for the source side of the training data using the sent2vec model 'sent2vec.model.bin') <br>
```
nfr-create-faiss-index  -c ./0_preprocessing/bpe_merged/train.tok.truec.bpe.en --model_name_or_path ./sent2vec/sent2vec.train.tok.truec.10dim.bpe.bin -o ./sent2vec/sent2vec.faiss.en
```

To extract fuzzy matches (for the training set): <br>
```
nfr-extract-fuzzy-matches --tmsrc ./0_preprocessing/bpe_merged/train.tok.truec.bpe.en --tmtgt ./0_preprocessing/bpe_merged/train.tok.truec.bpe.nl --insrc ./0_preprocessing/bpe_merged/train.tok.truec.bpe.en --method sent2vec --faiss ./sent2vec/sent2vec.faiss.10dim.en --model_name_or_path ./sent2vec/sent2vec.train.tok.truec.10dim.bpe.bin --maxmatch 40 --minscore 0.5 --threads 1
```

***Note 1:** This command generates the 'fuzzy match file' (train.tok.truec.bpe.en.matches.mins0.5.maxm40.sent2vec.txt) in the same folder as the original file (`--insrc`).* <br>
***Note 2:** Modify `--insrc` parameter to extract fuzzy matches for the development or test sets separately.* <br>
***Note 3:** To run the process on GPU, remove `--threads` parameter and use `--use_cuda` instead.*

### Step 2. Add (NMT) Training Features 
Features are added to fuzzy match files for the training, test and development sets separately. <br>
This step requires a word alignment file in Pharaoh format (only for the training set!), where the the source and target token indices are separated by a dash. For instance: 0-0 1-1 2-2 2-3 3-4 4-5 <br>
We used [GIZA++](https://github.com/moses-smt/giza-pp) to obtain alignments for the training set (tokenized, truecased, byte-pair encoded). Please see the paper for the parameters we used for GIZA++.

To add features to the fuzzy match file of the training set: <br>
```
nfr-add-training-features ./1_fuzzy_matches/train.tok.truec.bpe.en.matches.mins0.5.maxm40.sent2vec.txt ./word_alignments/train.bpe.alignments -l
```

***Note 1:** This command generates a new 'fuzzy match file' with features (train.tok.truec.bpe.en.matches.mins0.5.maxm40.sent2vec.trainfeats.txt) in the same folder as the input file (`--insrc`).* <br>
***Note 2:** Change the first positional argument to add features to the fuzzy match files for the development or test set separately.* <br>
***Note 3:** Only the word alignment file obtained on the training set is used (also for adding features to test or dev sets)!* <br>
***Note 4:** In this step, both "side (source/target)" and "match/nomatch" features are added to the fuzzy matche files. While in the final augmented data we do not use the "match/nomatch" features on the source tokens, this feature is still necessary to apply "max. coverage" during the data augmentation step (Step 3).*

### Step 3. Augment Data

To augment training set: <br>
```
nfr-augment-data --src ./0_preprocessing/bpe_merged/train.tok.truec.bpe.en --tgt ./0_preprocessing/bpe_merged/train.tok.truec.bpe.nl --fm ./2_fuzzy_matches_w_features/train.tok.truec.bpe.en.matches.mins0.5.maxm40.sent2vec.trainfeats.txt --minscore 0.5 --n_matches 2 -sf side -ftf matched --outdir ./3_augment_data/train/ --combine max_coverage --is_trainset
```

***Note 1:** When augmenting the training set we need to use the `--is_trainset` parameter as the training set is augmented in a different way compared to the test and dev sets. Please see the paper for details on data augmentation.* <br>
***Note 2:** This command creates the augmented training set (source/target) in the output directory, which can be used to train the NMT model.* <br>


To augment test (or dev) set: <br>
```
nfr-augment-data --src ./0_preprocessing/bpe_merged/test.tok.truec.bpe.en --tgt ./0_preprocessing/bpe_merged/test.tok.truec.bpe.nl --fm ./2_fuzzy_matches_w_features/test.tok.truec.bpe.en.matches.mins0.5.maxm40.sent2vec.trainfeats.txt --minscore 0.5 --n_matches 2 -sf side -ftf matched --outdir ./3_augment_data/test/ --combine max_coverage
```

***Note 1:** Modify `--src`, `--tgt`, `--fm` and the output directory (`--outdir`) to augment the dev set.* <br>
***Note 2:** This command creates the augmented test/dev set (source/target) in the output directory, which can be used to translate using the NMT model trained on augmented training set.* <br>
***Note 3:** If you want to generate documents containing sentences per fuzzy match range, add the parameter `--out_ranges`.*

### Step 4. Train NMT model
In our experiments we used OpenNMT-py (version 1.x) to train NMT models but the augmented data sets can be used to train models with any toolkit provided that it supports source-side features (OpenNMT-py version 2 does not support source-side features at the time of writing this guide).
Some parameters to pay attention to during the "preprocessing" step in OpenNMT:
- `onmt_preprocess`: `src-seq_length` is increased based on the no. of fuzzy matches used for data augmentation (x2 when a single fuzzy match is used; x3 when 2 fuzzy matches are used etc.)
- `onmt_train`: `word_vec_size` + `feat_vec_size` = `rnn_size`. For ex. `word_vec_size` = 506, `feat_vec_size` = 6 and `rnn_size` = 512. If The sum of `word_vec_size` and `feat_vec_size` is not equal to `rnn_size` it gives an error ([source](https://github.com/OpenNMT/OpenNMT-py/issues/1534)).

Please see the paper for a list of all the paramater values we used in our experiments.

 
## Citation

Please cite [our paper(s)](CITATION) when you use this library.
 
If you perform automatic or manual evaluations or analyse how similar translations influence the MT output:

Tezcan, A., Bulté, B. (2022). Evaluating the Impact of Integrating Similar Translations into Neural Machine Translation. *Informatics*, 13(1). https://www.mdpi.com/2078-2489/13/1/19

```
@Article{info13010019,
AUTHOR = {Tezcan, Arda and BultÃ©, Bram},
TITLE = {Evaluating the Impact of Integrating Similar Translations into Neural Machine Translation},
JOURNAL = {Information},
VOLUME = {13},
YEAR = {2022},
NUMBER = {1},
ARTICLE-NUMBER = {19},
URL = {https://www.mdpi.com/2078-2489/13/1/19},
ISSN = {2078-2489},
ABSTRACT = {Previous research has shown that simple methods of augmenting machine translation training data and input sentences with translations of similar sentences (or fuzzy matches), retrieved from a translation memory or bilingual corpus, lead to considerable improvements in translation quality, as assessed by a limited set of automatic evaluation metrics. In this study, we extend this evaluation by calculating a wider range of automated quality metrics that tap into different aspects of translation quality and by performing manual MT error analysis. Moreover, we investigate in more detail how fuzzy matches influence translations and where potential quality improvements could still be made by carrying out a series of quantitative analyses that focus on different characteristics of the retrieved fuzzy matches. The automated evaluation shows that the quality of NFR translations is higher than the NMT baseline in terms of all metrics. However, the manual error analysis did not reveal a difference between the two systems in terms of total number of translation errors; yet, different profiles emerged when considering the types of errors made. Finally, in our analysis of how fuzzy matches influence NFR translations, we identified a number of features that could be used to improve the selection of fuzzy matches for NFR data augmentation.},
DOI = {10.3390/info13010019}
}
```



If you use semantic fuzzy matching (sent2vec, sentence-transformers), sub-word segmentation, max. coverage for combining fuzzy matches, source-side features for training NMT models:

Tezcan, A., Bulté, B., & Vanroy, B. (2021). Towards a better integration of fuzzy matches in neural machine
 translation through data augmentation. *Informatics*, 8(1). https://doi.org/10.3390/informatics8010007

```
@article{tezcan2021integration,
    AUTHOR = {Tezcan, Arda and Bulté, Bram and Vanroy, Bram},
    TITLE = {Towards a Better Integration of Fuzzy Matches in Neural Machine Translation through Data Augmentation},
    JOURNAL = {Informatics},
    VOLUME = {8},
    YEAR = {2021},
    NUMBER = {1},
    ARTICLE-NUMBER = {7},
    URL = {https://www.mdpi.com/2227-9709/8/1/7},
    ISSN = {2227-9709},
    ABSTRACT = {We identify a number of aspects that can boost the performance of Neural Fuzzy Repair (NFR), an easy-to-implement method to integrate translation memory matches and neural machine translation (NMT). We explore various ways of maximising the added value of retrieved matches within the NFR paradigm for eight language combinations, using Transformer NMT systems. In particular, we test the impact of different fuzzy matching techniques, sub-word-level segmentation methods and alignment-based features on overall translation quality. Furthermore, we propose a fuzzy match combination technique that aims to maximise the coverage of source words. This is supplemented with an analysis of how translation quality is affected by input sentence length and fuzzy match score. The results show that applying a combination of the tested modifications leads to a significant increase in estimated translation quality over all baselines for all language combinations.},
    DOI = {10.3390/informatics8010007}
}
```

If you use lexical fuzzy matching (editdist, setsim, setsimeditdist):

Bulte, B., & Tezcan, A. (2019). Neural Fuzzy Repair: Integrating Fuzzy Matches into Neural Machine
 Translation. In *Proceedings of the 57th Annual Meeting of the Association for Computational
 Linguistics* (pp. 1800–1809). Association for Computational Linguistics. https://www.aclweb.org/anthology/P19-1175

```
@inproceedings{bulte2019neural,
    AUTHOR = {Bulte, Bram  and Tezcan, Arda},
    TITLE = {Neural Fuzzy Repair: Integrating Fuzzy Matches into Neural Machine Translation},
    BOOKTITLE = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    MONTH = jul,
    YEAR = {2019},
    ADDRESS = {Florence, Italy},
    PUBLISHER = {Association for Computational Linguistics},
    URL = {https://www.aclweb.org/anthology/P19-1175},
    PAGES = {1800--1809},
    ABSTRACT = {We present a simple yet powerful data augmentation method for boosting Neural Machine Translation (NMT) performance by leveraging information retrieved from a Translation Memory (TM). We propose and test two methods for augmenting NMT training data with fuzzy TM matches. Tests on the DGT-TM data set for two language pairs show consistent and substantial improvements over a range of baseline systems. The results suggest that this method is promising for any translation environment in which a sizeable TM is available and a certain amount of repetition across translations is to be expected, especially considering its ease of implementation.},
    DOI = {10.18653/v1/P19-1175},
}
```

## Development
After larger refactors and before new releases, always run the following commands in the root of the current directory
 to ensure a consistent code style. We use additional plugins to help with that. They can automatically be installed
 by using the `dev` option.

```bash
pip install .[dev]
```

For a consistent **coding style**, the following command will reformat the files in `nfr/`.

```bash
make style
```

We use black-style conventions alongside isort.

For **quality checking** we use flake8. Run the following command and make sure to fix all warnings. Only
 publish a new release when no more warnings or errors are present.

```bash
make quality
```
