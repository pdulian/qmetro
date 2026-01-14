from setuptools import setup, find_packages




with open('README.md', 'r') as f:
    description = f.read()

setup(
    name='qmetro',
    version='1.1.0',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.26.4',
        'scipy>=1.14.1',
        'cvxpy>=1.6.0',
        'matplotlib>=3.10.0',
        'networkx>=3.3',
        'ncon',
    ],
    long_description=description,
    long_description_content_type='text/markdown',
)
