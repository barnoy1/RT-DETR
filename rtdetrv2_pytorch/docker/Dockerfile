FROM ghcr.io/astral-sh/uv:latest AS uv

FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /workspace

ARG PIP_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN set -eux; \
    if [ -d /etc/apt/sources.list.d ]; then \
        find /etc/apt/sources.list.d -type f -name '*.list' \
            -exec sed -i '/developer\.download\.nvidia\.com/s/^/# temporarily disabled: /' {} + || true; \
    fi; \
    apt-get -o Acquire::ForceIPv4=true \
            -o Acquire::Retries=5 \
            -o Acquire::http::Timeout=30 \
            -o Acquire::https::Timeout=30 \
            -o Acquire::http::No-Cache=true \
            -o Acquire::PDiffs=false \
            update; \
    apt-get install -y --no-install-recommends \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcairo2 \
    libdbus-1-3 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxkbfile1 \
    libxrandr2 \
    xdg-utils \
    xz-utils; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

COPY --from=uv /uv /uvx /usr/local/bin/

RUN uv --version && \
    uv venv --system-site-packages "$VIRTUAL_ENV" && \
    "$VIRTUAL_ENV/bin/python" -c "import torch; print('Detected base torch:', torch.__version__)" && \
    grep -Eiv '^[[:space:]]*(torch|torchvision)([[:space:]]*[<>=!~].*)?[[:space:]]*$' requirements.txt > /tmp/requirements.docker.txt && \
    uv pip install --python "$VIRTUAL_ENV/bin/python" -r /tmp/requirements.docker.txt

COPY --from=host_programs . /opt/programs/
COPY scripts/* /usr/local/bin/
RUN set -eux; \
    chmod +x /usr/local/bin/launch_vscode.sh; \
        VSCODE_DEBS="$(find /opt/programs -maxdepth 3 -type f \( -name 'vscode-*_amd64.deb' -o -name 'code_*_amd64.deb' \) | sort || true)"; \
        if [ -n "$VSCODE_DEBS" ]; then \
            for deb in $VSCODE_DEBS; do \
                dpkg -i "$deb" || true; \
            done; \
            apt-get -o Acquire::ForceIPv4=true \
                    -o Acquire::Retries=5 \
                    -o Acquire::http::Timeout=30 \
                    -o Acquire::https::Timeout=30 \
                    update; \
            apt-get install -y --no-install-recommends -f; \
            rm -rf /var/lib/apt/lists/*; \
        else \
            echo "Warning: no VS Code .deb package found under /opt/programs; skipping dpkg install."; \
        fi; \
        if [ -x /usr/share/code/bin/code ]; then \
            ln -sfn /usr/share/code /opt/vscode; \
        fi; \
        if [ ! -e /opt/vscode ]; then \
            VSCODE_BIN="$(find /opt/programs -maxdepth 8 -type f \( -path '*/bin/code' -o -name 'code' \) | head -n1 || true)"; \
            if [ -n "$VSCODE_BIN" ]; then \
                if [ "$(basename "$(dirname "$VSCODE_BIN")")" = "bin" ]; then \
                    VSCODE_DIR="$(dirname "$(dirname "$VSCODE_BIN")")"; \
                else \
                    VSCODE_DIR="$(dirname "$VSCODE_BIN")"; \
                fi; \
                ln -sfn "$VSCODE_DIR" /opt/vscode; \
            else \
                echo "Warning: VS Code binary not discovered at build time under /opt/programs; launcher will try runtime discovery."; \
            fi; \
        fi; \
    ln -sf /usr/local/bin/launch_vscode.sh /usr/local/bin/code; \
    ln -sf /usr/local/bin/launch_vscode.sh /usr/bin/code; \
    mkdir -p /root/.vscode/extensions; \
    find /opt/programs -maxdepth 12 -type d -path '*/.vscode/extensions/*' -print0 2>/dev/null \
        | xargs -0 -r -I{} sh -c 'if [ -f "$1/package.json" ]; then cp -a "$1" /root/.vscode/extensions/; fi' _ {}; \
    if [ -x /usr/share/code/bin/code ]; then \
        find /opt/programs -maxdepth 12 -type f -name '*.vsix' -print0 2>/dev/null \
            | xargs -0 -r -I{} sh -c '/usr/share/code/bin/code --no-sandbox --user-data-dir /tmp/vscode-root --extensions-dir /root/.vscode/extensions --install-extension "$1" --force || true' _ {}; \
        if [ -f /opt/programs/extensions.txt ]; then \
            sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' /opt/programs/extensions.txt \
                | xargs -r -I{} /usr/share/code/bin/code --no-sandbox --user-data-dir /tmp/vscode-root --extensions-dir /root/.vscode/extensions --install-extension {} --force || true; \
        fi; \
    fi

CMD ["/bin/bash"]