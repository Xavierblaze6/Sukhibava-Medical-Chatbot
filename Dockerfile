FROM python:3.10-slim

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python ingest.py

RUN mkdir -p /home/user/app/.files && \
	chown -R user:user /home/user/app

USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME=/home/user

EXPOSE 7860

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]