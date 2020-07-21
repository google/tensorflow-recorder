init:
	pip install -r requirements.txt

test:
	nosetests --with-coverage --nocapture -v --cover-package=tfrecorder

pylint:
	pylint tfrecorder

.PHONY: init glint coverage test
