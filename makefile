
init:
	python -m venv .venv

install_deps:
	.venv/bin/pip install -r requirements.txt

python:
	.venv/bin/python

typecheck:
	.venv/bin/mypy pagerank tests

black:
	.venv/bin/black --check pagerank tests

test:
	.venv/bin/python -m pytest -x -v --pdb --cov=pagerank --cov-report term-missing

lint:
	.venv/bin/pylint pagerank tests
