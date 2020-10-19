all: init test pylint

init:
	pip install -r requirements.txt

test:
	nosetests --with-coverage -v --cover-package=tfrecorder

pylint:
	pylint -j 0 tfrecorder

.PHONY: all init test pylint 
