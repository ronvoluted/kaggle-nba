FROM python:3.9.1

WORKDIR /home/kaggle

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install pipenv && pipenv install --ignore-pipfile

COPY [".", "./"]

CMD ["pipenv", "run", "jupyter", "lab", "--allow-root", "--ip", "0.0.0.0"]
