init:
	pip install -r requirements.txt

test:
	python -m unittest

coverage:
	coverage run -m unittest && coverage report -m

lint:
	glint tfrutil/*.py

.PHONY: init lint coverage test