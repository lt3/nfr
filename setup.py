from pathlib import Path
from setuptools import find_packages, setup

from nfr import __version__

extras = {"dev": ["isort>=5.5.4", "black", "flake8", "pygments"]}

setup(
    name="nfr",
    version=__version__,
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["nfr", "nfr.*"]),
    license="Apache 2.0",
    author="Arda Tezcan, Bram Vanroy",
    author_email="arda_te@yahoo.com",
    url="https://github.com/lt3/nfr/",
    project_urls={
        "Bug Reports": "https://github.com/lt3/nfr/issues",
        "Source": "https://github.com/lt3/nfr",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.6",
    install_requires=[
        "editdistance",
        "nltk",
        "numpy",
        "pandas",
        "setsimilaritysearch==0.1.7",
        "tqdm",
    ],
    extras_require=extras,
    entry_points={
        "console_scripts": ["nfr-add-training-features=nfr.add_training_features:main",
                            "nfr-augment-data=nfr.augment_data:main",
                            "nfr-create-faiss-index=nfr.fuzzy_matching.create_faiss_index:main",
                            "nfr-extract-fuzzy-matches=nfr.fuzzy_matching.extract_fuzzy_matches:main"]
    }
)
