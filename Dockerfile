FROM python:3.7.3-stretch

## Step 1:
WORKDIR /project

## Step 2:
COPY ./app.py /project/
COPY ./requirements.txt /project/
COPY ./model.h5 /project/

## Step 3:
RUN pip3 install -r requirements.txt
# hadolint ignore=DL3013

## Step 4:
EXPOSE 80
## Step 5:
ENTRYPOINT  ["python3","app.py"]
