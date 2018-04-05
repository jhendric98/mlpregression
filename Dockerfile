FROM ubuntu:latest
MAINTAINER Jim Hendricks "jhendric98@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential sqlite
COPY . /python_rest
WORKDIR /python_rest
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["server.py"]
