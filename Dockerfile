FROM continuumio/anaconda3:4.4.0
COPY ./mlaas /usr/local/python/
EXPOSE 5000
WORKDIR /usr/local/python/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python flask_predict_api.py.py