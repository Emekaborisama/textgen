from setuptools import find_packages, setup
setup(
    name='textgen',
    packages=find_packages(include=['textgen']),
    version = '0.1.0',
    description='build a text generation model',
    #long_description_content_type = 'Tensor text generation is a python library that allow you to build a text generation model',
    author='Emeka Boris Ama',
    author_email = 'borisphilosophy@gmail.com',
    license='MIT',
    url='https://github.com/Emekaborisama/textgen',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)

