# syntax=docker/dockerfile:1
FROM python:3.8

# Streamlit listens on 8501 by default; many platforms prefer 8080.
EXPOSE 8080
WORKDIR /app
COPY . ./
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     STREAMLIT_SERVER_PORT=8080     STREAMLIT_SERVER_ADDRESS=0.0.0.0     STREAMLIT_SERVER_ENABLE_CORS=false     STREAMLIT_BROWSER_GATHER_USAGE_STATS=false


# Install dependencies first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the app
COPY . /app

# Run
CMD ["streamlit", "run", "streamlit_app.py"]
