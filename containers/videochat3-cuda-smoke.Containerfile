ARG CUDA_IMAGE=docker.io/pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
FROM ${CUDA_IMAGE}

WORKDIR /workspace

COPY configs/vqa/videochat3_cuda_requirements.txt /workspace/configs/vqa/videochat3_cuda_requirements.txt
COPY scripts/bootstrap-videochat3-cuda-smoke.sh /workspace/scripts/bootstrap-videochat3-cuda-smoke.sh

RUN bash /workspace/scripts/bootstrap-videochat3-cuda-smoke.sh \
    /workspace/configs/vqa/videochat3_cuda_requirements.txt
