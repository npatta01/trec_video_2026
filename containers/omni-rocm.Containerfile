FROM localhost/trec-vqa-gate-gemma@sha256:5bbd6f039622f70ad53d62395f189b07dc8fe28e2336c8aa462081abee9b7aaa

RUN python -m pip install --no-cache-dir --only-binary=:all: \
      backoff==2.2.1 \
      peft==0.13.2 \
      timm==1.0.28 \
    && python -c "import backoff, peft, timm; print(backoff.__version__, peft.__version__, timm.__version__)"
