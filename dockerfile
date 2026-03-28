FROM  python:3.9-alpine
COPY . /app
WORKDIR /app
RUN pip install -r d:\deployement\requirements.txt
CMD python application.py