# Format source code automatically
style:
	black --line-length 119 --target-version py36 nfr
	isort nfr

# Control quality
quality:
	black --check --line-length 119 --target-version py36 nfr
	isort --check-only nfr
	flake8 nfr --exclude __pycache__,__init__.py
