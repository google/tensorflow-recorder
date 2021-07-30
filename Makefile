all: init test-nb test pylint

init:
	pip install -r requirements.txt

test: test-nb test-py

test-py:
	nosetests --with-coverage -v --cover-package=tfrecorder

test-nb:
	ls -1 samples/*.ipynb | grep -v '^.*Dataflow.ipynb' | xargs py.test --nbval-lax -p no:python

pylint:
	pylint -j 0 tfrecorder

.PHONY: all init test pylint
