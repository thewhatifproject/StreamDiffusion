import re

from setuptools import find_packages, setup

_deps = [
    "diffusers==0.31.0",
    "transformers",
    "accelerate",
    "fire",
    #"omegaconf",
    "cuda-python",
    "onnx==1.17.0",
    "onnxruntime==1.20.1",
    "onnxruntime-gpu",
    "onnx_graphsurgeon>=0.5.2",
    "protobuf==3.20.2",
    "colored",
    "pywin32;sys_platform == 'win32'",
    "controlnet-aux==0.0.9",
    "huggingface_hub==0.25.0",
    "numpy==1.26.4",
    "peft==0.6.0",
    "polygraphy",
    "compel"
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]

extras = {}
extras["cuda"] = deps_list("protobuf", "cuda-python", "onnx", "onnxruntime", "colored")
extras["cuda-tensorrt"] = deps_list("protobuf", "cuda-python", "onnx", "onnxruntime", "onnxruntime-gpu", "colored", "polygraphy", "onnx_graphsurgeon")

install_requires = [
    deps["fire"],
    #deps["omegaconf"],
    deps["diffusers"],
    deps["transformers"],
    deps["accelerate"],
    deps["controlnet-aux"],
    deps["huggingface_hub"],
    deps["numpy"],
    deps["peft"],
    deps["compel"]
]

setup(  
    name="streamdiffusion",
    version="0.1.0",
    description="real-time interactive image generation pipeline",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning diffusion pytorch stable diffusion audioldm streamdiffusion real-time",
    license="Apache 2.0 License",
    author="Aki, kizamimi, ddPn08, Verb, ramune, teftef6220, Tonimono, Chenfeng Xu, Ararat with the help of all our contributors (https://github.com/cumulo-autumn/StreamDiffusion/graphs/contributors)",
    author_email="cumulokyoukai@gmail.com",
    url="https://github.com/cumulo-autumn/StreamDiffusion",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"streamdiffusion": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=list(install_requires),
    extras_require=extras,
)
