# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG ROBOCLAW_DOCKER_PROFILE=ubuntu2404-ros2
ARG ROBOCLAW_INSTALL_ROS2=0
ARG ROBOCLAW_ROS2_DISTRO=none
ARG ROBOCLAW_ROS2_CONTROL_PYTHON=/usr/local/bin/python3
ARG ROBOCLAW_PYTHON_VERSION=3.11
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG ALL_PROXY=
ARG http_proxy=
ARG https_proxy=
ARG all_proxy=
ENV ROBOCLAW_ROS2_DISTRO=${ROBOCLAW_ROS2_DISTRO}
ENV ROBOCLAW_ROS2_CONTROL_PYTHON=${ROBOCLAW_ROS2_CONTROL_PYTHON}
ENV ROBOCLAW_ROS2_CONTROL_PYTHONPATH=/usr/local/lib/python${ROBOCLAW_PYTHON_VERSION}/dist-packages:/app
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV ALL_PROXY=${ALL_PROXY}
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV all_proxy=${all_proxy}

# Install Python 3.11 on both Ubuntu profiles and Node.js 20 for the WhatsApp bridge.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      dirmngr \
      git \
      gnupg \
      locales \
      lsb-release \
      python3-pip \
      software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      "python${ROBOCLAW_PYTHON_VERSION}" \
      "python${ROBOCLAW_PYTHON_VERSION}-venv" && \
    "python${ROBOCLAW_PYTHON_VERSION}" -m ensurepip --upgrade && \
    ln -sf "/usr/bin/python${ROBOCLAW_PYTHON_VERSION}" /usr/local/bin/python && \
    ln -sf "/usr/bin/python${ROBOCLAW_PYTHON_VERSION}" /usr/local/bin/python3 && \
    ln -sf "/usr/local/bin/pip${ROBOCLAW_PYTHON_VERSION}" /usr/local/bin/pip && \
    ln -sf /usr/local/bin/pip /usr/local/bin/pip3 && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN if [ "${ROBOCLAW_INSTALL_ROS2}" = "1" ]; then \
      locale-gen en_US en_US.UTF-8 && \
      update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
      curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | gpg --dearmor -o /etc/apt/keyrings/ros-archive-keyring.gpg && \
      . /etc/os-release && \
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${VERSION_CODENAME} main" > /etc/apt/sources.list.d/ros2.list && \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        "ros-${ROBOCLAW_ROS2_DISTRO}-ros-base" \
        python3-argcomplete \
        python3-colcon-common-extensions && \
      rm -rf /var/lib/apt/lists/*; \
    fi

LABEL roboclaw.docker_profile="${ROBOCLAW_DOCKER_PROFILE}"
LABEL roboclaw.ros2_distro="${ROBOCLAW_ROS2_DISTRO}"

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p roboclaw bridge && touch roboclaw/__init__.py && \
    python -m pip install --no-cache-dir --break-system-packages --ignore-installed . && \
    rm -rf roboclaw bridge

# Copy the full source and install
COPY roboclaw/ roboclaw/
COPY bridge/ bridge/
RUN python -m pip install --no-cache-dir --break-system-packages --ignore-installed .
RUN if [ "${ROBOCLAW_INSTALL_ROS2}" = "1" ]; then \
      /usr/bin/python3 -m pip install --no-cache-dir --break-system-packages --ignore-requires-python --ignore-installed . \
        || (curl -fsSL https://bootstrap.pypa.io/get-pip.py | /usr/bin/python3 - --break-system-packages --force-reinstall && \
            /usr/bin/python3 -m pip install --no-cache-dir --break-system-packages --ignore-requires-python --ignore-installed .); \
    fi
RUN python -m pip install --no-cache-dir --break-system-packages mujoco Pillow || true
RUN python -c "import roboclaw; import scservo_sdk; print('scservo_sdk: found (vendored)')"
RUN if [ "${ROBOCLAW_INSTALL_ROS2}" = "1" ]; then \
      /usr/bin/python3 -c "import roboclaw; import scservo_sdk; import pydantic; import pydantic_core; print('control-python modules: ok')"; \
    fi

RUN mv /usr/local/bin/roboclaw /usr/local/bin/roboclaw-real
COPY scripts/docker/roboclaw-wrapper.sh /usr/local/bin/roboclaw
RUN chmod +x /usr/local/bin/roboclaw

# Build the WhatsApp bridge
WORKDIR /app/bridge
RUN npm install && npm run build
WORKDIR /app

# Clear build-time proxy defaults from the final image. Runtime proxy values are
# injected by the Docker workflow scripts when needed.
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
ENV ALL_PROXY=
ENV http_proxy=
ENV https_proxy=
ENV all_proxy=

# Create config directory
RUN mkdir -p /root/.roboclaw

# Gateway default port
EXPOSE 18790

ENTRYPOINT ["roboclaw"]
CMD ["status"]
