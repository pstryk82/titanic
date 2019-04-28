FROM python:3.6

ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt

RUN pip install -e .

WORKDIR /code
CMD ["python", "app/classify.py"]

CMD ["bash"]