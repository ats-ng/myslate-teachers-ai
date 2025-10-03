# gunicorn_config.py (minimal version)
import multiprocessing

bind = "127.0.0.1:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
max_requests = 1000
max_requests_jitter = 100
timeout = 120
keepalive = 2
accesslog = "-"
errorlog = "-"
loglevel = "info"