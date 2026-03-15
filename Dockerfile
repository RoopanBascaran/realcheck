FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gosu \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Nahrawy/AIorNot model at build time
RUN python -c "from transformers import AutoImageProcessor, AutoModelForImageClassification; AutoImageProcessor.from_pretrained('Nahrawy/AIorNot'); AutoModelForImageClassification.from_pretrained('Nahrawy/AIorNot')"

COPY --chown=user:user . .

# Switch to root for entrypoint (configures DNS, then drops to user)
USER root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/entrypoint.sh"]
