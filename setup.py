from setuptools import setup, find_packages
import sys, os.path

setup(name='aiagents',
      version='0.1',
      description='AI agents and multi-agent algorithms.',
      url='https://github.com/INFLUENCEorg/aiagents',
      author='Influence TEAM',
      author_email='author@example.com',
      license='Example',
      packages=['aiagents', 'aiagents/single', 'aiagents/single/PPO', 'aiagents/single/mcts', 'aiagents/multi'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['absl-py', 'astor', 'atari-py', 'certifi', 'chardet', 'cycler', 'future', 'gast', 'grpcio', 'gym', 'h5py', 'idna', 'Keras-Applications', 'Keras-Preprocessing', 'kiwisolver', 'Markdown', 'matplotlib', 'mock', 'numpy', 'pkg-resources', 'protobuf', 'pyglet', 'pyparsing', 'python-dateutil', 'PyYAML', 'requests', 'scipy', 'six', 'tensorboard', 'tensorflow', 'tensorflow-estimator', 'termcolor', 'urllib3', 'Werkzeug'
        ]
)
