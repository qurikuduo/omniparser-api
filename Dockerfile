#FROM registry.hf.space/microsoft-omniparser:latest
FROM registry.hf.space/microsoft-omniparser@sha256:604fc53b4a66545c0723d6c0f8711e7321cd9eb9600a06156fe3eee9d8a54e92

USER root
RUN pip install  numpy==1.26.4 --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir paddleocr==2.7.3 paddlepaddle==2.6.2 paddlepaddle-gpu==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN chmod 1777 /tmp \
    && apt update -q && apt install -y ca-certificates wget libgl1 \
    && wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb && apt update -q \
    && apt install -y --no-install-recommends libcudnn8 libcublas-12-2

RUN pip install fastapi[all] 
RUN pip install  "numpy==1.26.4" --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY main.py main.py
RUN python main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
