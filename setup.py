from setuptools import setup, find_packages


setup(name='nlpete',
      description='An open-source NLP research library, built on PyTorch within the AllenNLP framework',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      url='https://github.com/epwalsh/nlp-models',
      author='Evan Pete Walsh',
      author_email='epwalsh10@gmail.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      python_requires='>=3.6.1')
