init:
	pip install -r requirements.txt

test:
	nosetests --with-coverage --nocapture -v --cover-package=tfrutil

pylint:
	pylint tfrutil

glint:
	glint tfrutil/*.py

.PHONY: init glint coverage test