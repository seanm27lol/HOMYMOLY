FROM nvcr.io/nvidia/pytorch:26.06-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir \
      "networkx>=3.3,<4" \
      "tensorboard>=2.18,<3" \
    && python -m pip install --no-cache-dir --no-deps --ignore-installed \
      "PyYAML>=6.0.2,<7" \
    && python -c "from packaging.version import Version; import numpy, torch; assert (2, 8) <= Version(torch.__version__.split('+')[0]).release[:2] < (3, 0); assert 2 <= Version(numpy.__version__).major < 3" \
    && python -m pip install --no-cache-dir --no-deps .

COPY configs ./configs
COPY scripts ./scripts
COPY docs ./docs
RUN chmod +x \
    /workspace/scripts/audit_shortcuts.py \
    /workspace/scripts/gpu_idle_train.py \
    /workspace/scripts/install_training_cron.py \
    /workspace/scripts/profile_gb10.py \
    /workspace/scripts/start_dashboard.sh \
    /workspace/scripts/train_gate2.sh

ENTRYPOINT ["python", "-m", "homymoly"]
CMD ["check-config", "--config", "/workspace/configs/stage1.yaml", "--create-artifacts"]
