# -*- coding: utf-8 -*-
from setuptools import setup, find_packages




with open('README.md', 'r') as f:
    description = f.read()

setup(
    name='qmetro',
    version='1.1.1',
    description=(
        'A package that provides a set of tools for identifying '
        'optimal estimation protocols that maximize quantum Fisher '
        'information (QFI).'),
    url='https://github.com/pdulian/qmetro',
    project_urls={
        'Documentation': 'https://qmetro.readthedocs.io/en/latest/',
        'Source': 'https://github.com/pdulian/qmetro',
        'Article': 'https://arxiv.org/abs/2506.16524'
    },
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
    author='Piotr Dulian and Stanisław Kurdziałek',
    author_email='p.dulian@cent.uw.edu.pl',
    license='GPLv3',
    keywords='quantum metrology optimization physics',
)
