from chutes.image import Image


chute_docker_image = (
    Image(username="user", name="tripo-sg-mesh-generator", tag="1.0")
    .from_base("nvidia/cuda:12.8-runtime-ubuntu22.04")
    .with_python("3.11")

    # installing system libraries
    .run_command("""
        apt-get update && apt-get install -y \\
        git curl wget \\
        && rm -rf /var/lib/apt/lists/*
    """)
    .run_command("apt-get install -y cmake ninja")

    # installing python packages
    .run_command("pip install --upgrade pip")
    .run_command("pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128")
    .run_command("""
        pip install \\
        diffusers==0.35.1 \\ 
        transformers==4.56.2 \\
        einops==0.8.1 \\
        huggingface_hub==0.35.3 \\
        hf_transfer==0.1.9 \\
        opencv-python==4.12.0.88 \\
        trimesh==4.8.3 \\
        omegaconf==2.3.0 \\
        scikit-image==0.25.2 \\
        numpy==2.2.6 \\
        peft==0.17.1 \\
        jaxtyping==0.3.3 \\
        typeguard==4.4.4 \\
        diso==0.1.4 \\
        pymeshlab==2025.7 \\
        torchvision==0.22.1 \\
        matplotlib \\
        kornia==0.8.1 \\
        timm==1.0.20 \\
        diffrp-nvdiffrast \\
        tripo-sg @ git+https://github.com/404-Repo/tripo-sg-generator-chutes.git
    """)

    # Set up model cache directory
    .with_env("TRANSFORMERS_CACHE", "/app/models")
    .with_env("HF_HOME", "/app/models")
    .with_env("CUDA_VISIBLE_DEVICES", "0")
    .run_command("mkdir -p /app/models")

    # Set working directory
    .set_workdir("/app")
)