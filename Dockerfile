FROM continuumio/anaconda3:5.2.0
LABEL maintainer="Alex Lim https://www.linkedin.com/in/alexlim95"
COPY . /app
EXPOSE 5000
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]