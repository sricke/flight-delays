
# Get python image
FROM python:3.9

WORKDIR /code

# copy dependencies
COPY ./requirements.txt /code/requirements.txt

#ENV PORT 80

# install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# copy model  
COPY ./models/flight_delay.json /code/models/flight_delay.json

# copy api code
COPY ./app.py /code/app.py

# run app
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1