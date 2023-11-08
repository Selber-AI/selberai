FROM python:3

COPY . /selberai
WORKDIR /selberai

# install twine and build
RUN python3 -m pip install --upgrade twine build

# build
RUN python3 -m build

# First upload, then download and install to see if all works
CMD python3 -m twine upload --config-file .pypirc_test --repository testpypi dist/* \
    && python3 -m pip install -i https://test.pypi.org/simple/ selberai

