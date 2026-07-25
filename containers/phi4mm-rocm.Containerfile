FROM localhost/trec-vqa-gate-gemma@sha256:5bbd6f039622f70ad53d62395f189b07dc8fe28e2336c8aa462081abee9b7aaa

RUN python -m pip install --no-cache-dir --only-binary=:all: \
      accelerate==1.3.0 \
      backoff==2.2.1 \
      peft==0.13.2 \
      transformers==4.48.2 \
    && python -c "import accelerate, backoff, peft, transformers; print(accelerate.__version__, backoff.__version__, peft.__version__, transformers.__version__)"
