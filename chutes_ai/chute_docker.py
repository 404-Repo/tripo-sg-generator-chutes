from chutes.image import Image


chute_docker_image = (
    Image(username="", name="tripo-sg-mesh-generator", tag="0.0")
    .from_base("nvidia/cuda:12.8.0-devel-ubuntu24.04")

    # installing system libraries
    .run_command("apt update && apt install -y python3-full python3-dev python3-pip python3-wheel git build-essential libpq-dev")

    # installing python packages
    .run_command("ln -sf /usr/bin/pip3 /usr/local/bin/pip")
    .run_command("pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128 --break-system-packages")
    .run_command("CPATH='/usr/local/cuda/include:$CPATH' pip3 install diso==0.1.4 --break-system-packages")
    .run_command("pip install git+https://github.com/404-Repo/tripo-sg-generator-chutes.git --break-system-packages")
    .run_command("pip install --upgrade --break-system-packages chutes")

    # Set up model cache directory
    .run_command("mkdir -p /app/models")

    # Set working directory
    .set_workdir("/app")
)