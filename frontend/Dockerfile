FROM python:3.9.6

RUN mkdir /app
WORKDIR /app

COPY frontend/requirements.txt /app
COPY frontend/streamlit_app.py /app
COPY data/ /app/data

RUN pip install -r requirements.txt

ENV BACKEND_URL=http://backend:8000
ENV BASE_PATH=/app

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "streamlit_app.py"]