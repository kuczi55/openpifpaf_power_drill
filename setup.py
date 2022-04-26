from setuptools import setup, find_packages

setup(
    name="openpifpaf_power_drill",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["openpifpaf", 'numpy', 'torch', 'matplotlib', 'pandas'],
    extras_require={"dev": ["black"]},
)