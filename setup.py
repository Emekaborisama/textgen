from setuptools import find_packages, setup

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='text_gen',
    packages=find_packages(include=['text_gen']),
    version = '1.4.0',
    description='build a text generation model',
    long_description_content_type='text/markdown',
    long_description = README,
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

install_requires = ['pandas', 'tensorflow', 'torchvision', 'numpy', 'matplotlib', 'keras', 'setuptools>=47.1.1', 'parameter-sherpa', 'deepsegment', 'tensorflow-serving-api==1.12.0']

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
