from chutes.image import Image


class ChuteDockerImage:
    def __init__(self, username: str, docker_image_name: str, tag: str):
        self.chute_docker_image = (
            Image(username=username, name=docker_image_name, tag=tag)
            .from_base("nvidia/cuda:12.8.0-devel-ubuntu24.04")

            # installing system libraries
            .run_command("apt update -y && apt-get install software-properties-common -y && apt update -y \
                          && add-apt-repository ppa:deadsnakes/ppa -y && apt update -y")
            .run_command("apt install -y python3.11 python3-pip python3-wheel libcudnn9-cuda-12 \
                          gcc-14 g++-14 git build-essential libgl1 libglu1-mesa build-essential libpq-dev cmake ninja-build")
            .run_command("apt-get remove python3-cryptography -y")

            # rebinding names pip3 -> pip; python3 -> python; gcc-14 -> gcc; g++-14 -> g++
            .run_command("rm /usr/lib/python3*/EXTERNALLY-MANAGED")
            .run_command("update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 120 && \
                          update-alternatives --install /usr/bin/python python /usr/bin/python3 120 && \
                          update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 && \
                          update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100")

            .run_command("pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128")
            .run_command("CPATH='/usr/local/cuda/include:$CPATH' pip install git+https://github.com/404-Repo/tripo-sg-generator-chutes.git")

            # Set up model cache directory
            .run_command("mkdir -p /app/models")

            # Set working directory
            .set_workdir("/app")
            .add("./chute_docker.py", "/app/chute_docker.py")
            .add("./chute_io_data_structures.py", "/app/chute_io_data_structures.py")
        )