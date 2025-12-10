FROM registry.hf.space/microsoft-omniparser:latest

USER root
RUN pip install  numpy==1.26.4 --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir paddleocr==2.7.3 paddlepaddle==2.6.2 paddlepaddle-gpu==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN chmod 1777 /tmp \
    && apt update -q && apt install -y ca-certificates wget libgl1 \
    && wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb && apt update -q \
    && apt install -y --no-install-recommends libcudnn8 libcublas-12-2

RUN pip install fastapi[all] --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
# 关键修复：降级 Transformers 到 4.49.0（解决 _supports_sdpa 错误）
RUN pip install transformers==4.49.0 --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install  "numpy==1.26.4"  --no-cache-dir --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY main.py main.py
# 关键：构建时强制开网，只为下载 Florence-2 的 processor 代码和配置（只此一次）
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python main.py

# 构建完成后永久关闭网络（运行时彻底离线）
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
