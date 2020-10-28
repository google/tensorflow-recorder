all: init testnb test pylint

init:
	pip install -r requirements.txt

test:
	nosetests --with-coverage -v --cover-package=tfrecorder

testnb:
	ls -1 samples/*.ipynb | grep -v '^.*Dataflow.ipynb' | xargs py.test --nbval-lax -p no:python

pylint:
	pylint -j 0 tfrecorder

.PHONY: all init testnb test pylint
