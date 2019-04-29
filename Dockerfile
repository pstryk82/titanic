FROM reportbee/datascience:latest

ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt

RUN pip install -e .

RUN useradd -u 1000 user

WORKDIR /code
CMD ["python", "app/classify.py"]

CMD ["bash"]