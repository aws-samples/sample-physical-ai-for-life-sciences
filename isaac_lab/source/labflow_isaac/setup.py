from setuptools import find_packages, setup

setup(
    name="labflow-isaac",
    version="0.1.0",
    description="Isaac Lab DirectRLEnv for LabFlow Dreamer.",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
