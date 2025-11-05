import os
import sys
import random
import io
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import asyncio
from typing import Any, Union
from PIL import Image
from time import time
from pathlib import Path

import torch
import numpy as np
import trimesh
import pymeshlab
from huggingface_hub import snapshot_download
from chutes.chute import Chute, NodeSelector

from triposg.image_process import prepare_image
from triposg.briarmbg import BriaRMBG
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from chutes_ai.chute_docker import chute_docker_image
from chutes_ai.chute_io_data_structures import PipeInput, MeshOutput


# creating chute
chute = Chute(
    username="user",
    name="tripo-sg-generator",
    image=chute_docker_image,
    tagline="Tripo-SG 3D mesh AI generator",
    readme="""""",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24,
        include=["rtx4090"],
    ),
    concurrency=1
)

# initializing and downloading models
# @chute.on_startup()
async def load_model(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    triposg_path = Path("app/TripoSG")
    triposg_path.mkdir(parents=True, exist_ok=True)
    rmbg_path = Path("app/RMBG-1.4")
    rmbg_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_path.as_posix())
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_path.as_posix())

    # init rmbg model for background removal
    self.rmbg_net = BriaRMBG.from_pretrained(rmbg_path.as_posix()).to(self.device)
    self.rmbg_net.eval()

    # init tripoSG pipeline
    self.pipe = TripoSGPipeline.from_pretrained(triposg_path.as_posix()).to(self.device, torch.float16)

def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    return trimesh.Trimesh(vertices=verts, faces=faces)  #, vID, fID

def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
    if mesh.faces.shape[0] > n_faces:
        ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum = n_faces)
        return pymesh_to_trimesh(ms.current_mesh())
    else:
        return mesh

@torch.no_grad()
async def run_triposg(self,
    image_input: Union[str, Image.Image],
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1,
) -> tuple[trimesh.Trimesh, float]:

    t1 = time()
    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=self.rmbg_net)

    outputs = self.pipe(
        image=img_pil,
        generator=torch.Generator(device=self.pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).samples[0]
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if faces > 0:
        mesh = simplify_mesh(mesh, faces)
    t2 = time()
    total_time = (t2 - t1) / 60.0

    return mesh, total_time

@chute.cord(public_api_path="/generate", method="POST", input_schema=PipeInput)
async def generate_mesh(self, data: PipeInput) -> MeshOutput:
    seed = random.randint(0, 10000)
    image = Image.open(data.image_path)
    mesh, exec_time = await self.run_triposg(image=image, seed=seed, faces=data.num_faces)

    buffer = io.BytesIO()
    mesh.export(buffer)
    buffer.seek(0)

    return MeshOutput(
        mesh=buffer
    )


if __name__ == "__main__":
    async def test_locally():
        await load_model(chute)
        result = await generate_mesh(chute, PipeInput(image_path="/workspace/3148935_image_prompt.png", num_faces=-1))

    asyncio.run(test_locally())