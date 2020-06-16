init:
	pip install -r requirements.txt

test:
	nosetests --with-coverage --nocapture -v --cover-package=tfrutil

lint:
	glint tfrutil/*.py

.PHONY: init lint coverage test