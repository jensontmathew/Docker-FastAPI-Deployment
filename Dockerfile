FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt -vvv

CMD python app.py