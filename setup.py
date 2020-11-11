from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rigl-torch",
    version="0.5.2",
    author="Dyllan McCreary",
    author_email="mccreary@dyllan.ai",
    description="Implementation of Google Research's \"RigL\" sparse model training method in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/McCrearyD/rigl-torch",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
