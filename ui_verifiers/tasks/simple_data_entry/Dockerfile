FROM ubuntu:24.04

# Avoid interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install TigerVNC, a lightweight window manager (Openbox), Python 3, and minimal utilities
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tigervnc-standalone-server \
       tigervnc-common \
       tigervnc-tools \
       openbox \
       xterm \
       xdotool \
       xauth \
       dbus-x11 \
       ca-certificates \
       curl \
       procps \
       python3 \
       python3-xdg \
       python3-pip \
       python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    fastapi>=0.117.1 \
    pandas>=2.3.2 \
    pillow>=11.3.0 \
    playwright>=1.47 \
    python-xlib>=0.33 \
    screeninfo>=0.8.1 \
    uvicorn>=0.37.0

RUN playwright install-deps chromium

RUN useradd -ms /bin/bash ui-verifiers

ENV DISPLAY=:1

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

EXPOSE 5901

USER ui-verifiers
WORKDIR /home/ui-verifiers

RUN playwright install chromium

COPY *.py .

ENTRYPOINT ["/usr/local/bin/start.sh"]