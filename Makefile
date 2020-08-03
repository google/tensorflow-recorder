all: init pylint coverage test

init:
	pip install -r requirements.txt

test:
	nosetests --with-coverage --nocapture -v --cover-package=tfrecorder

pylint:
	pylint tfrecorder

.PHONY: all init test pylint 
