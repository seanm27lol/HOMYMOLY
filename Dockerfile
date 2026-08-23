ARG NGC_PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3
FROM ${NGC_PYTORCH_IMAGE}

ARG HOMYMOLY_INSTALL_MOLECULAR=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY constraints ./constraints
COPY src ./src
RUN python -m pip install --no-cache-dir \
      --constraint /workspace/constraints/gb10-ngc-26.06-direct.txt \
      networkx tensorboard \
    && python -m pip install --no-cache-dir --no-deps --ignore-installed \
      --constraint /workspace/constraints/gb10-ngc-26.06-direct.txt PyYAML \
    && if [ "${HOMYMOLY_INSTALL_MOLECULAR}" = "1" ]; then \
         python -m pip install --no-cache-dir \
           --constraint /workspace/constraints/gb10-ngc-26.06-direct.txt \
           ogb pandas rdkit; \
       fi \
    && python -c "from packaging.version import Version; import numpy, torch; assert (2, 8) <= Version(torch.__version__.split('+')[0]).release[:2] < (3, 0); assert 2 <= Version(numpy.__version__).major < 3" \
    && python -m pip install --no-cache-dir --no-deps .

COPY configs ./configs
COPY scripts ./scripts
COPY docs ./docs
RUN chmod +x \
    /workspace/scripts/audit_shortcuts.py \
    /workspace/scripts/export_artifact_bundle.py \
    /workspace/scripts/gpu_idle_train.py \
    /workspace/scripts/install_training_cron.py \
    /workspace/scripts/profile_gb10.py \
    /workspace/scripts/start_dashboard.sh \
    /workspace/scripts/train_gate2.sh

ENTRYPOINT ["python", "-m", "homymoly"]
CMD ["check-config", "--config", "/workspace/configs/stage1.yaml", "--create-artifacts"]
