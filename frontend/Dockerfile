FROM python:3.11-slim

WORKDIR /app

# install the frontend required packages
RUN pip install --no-cache-dir streamlit requests

# copy the Streamlit app
COPY frontend/frontend.py ./frontend.py

EXPOSE 8501

CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
