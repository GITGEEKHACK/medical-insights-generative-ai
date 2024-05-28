FROM python:3.9

WORKDIR /medical-insights

ADD ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip==23.3
RUN pip install -r requirements.txt
ADD ./app ./app
RUN pip install gunicorn
ADD run.py ./run.py
ADD ./conf.py ./conf.py

EXPOSE 8083
CMD gunicorn -c conf.py app:app