from setuptools import setup, find_packages
import sys, os.path

setup(name='aiagents',
      version='0.1',
      description='AI agents and multi-agent algorithms.',
      url='https://github.com/INFLUENCEorg/aiagents',
      author='Influence TEAM',
      author_email='author@example.com',
      license='Example',
      packages=['aiagents', 'aiagents/single', 'aiagents/single/PPO', 'aiagents/multi'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[]
)
