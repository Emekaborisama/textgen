from setuptools import find_packages, setup

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='text_gen',
    packages=find_packages(include=['text_gen']),
    version = '0.6.0',
    description='build a text generation model',
    #long_description_content_type = 'Tensor text generation is a python library that allow you to build a text generation model',
    author='Emeka Boris Ama',
    author_email = 'borisphilosophy@gmail.com',
    license='MIT',
    include_package_data=True,
    url='https://github.com/Emekaborisama/textgen',
    #install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)

install_requires = ['pandas', 'tensorflow', 'torchvision', 'numpy', 'matplotlib', 'keras', 'setuptools>=47.1.1', 'sherpa']


if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
