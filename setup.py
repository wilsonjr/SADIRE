from setuptools import find_packages, setup 

with open('README.md', 'r') as f:
	long_description = f.read()

setup(
	name='sadire',
	packages=find_packages(include=['sadire']),
	version='0.1.0',
	description='Sampling from scatter-plot visualizations',
	long_description=long_description,
    long_description_content_type='text/markdown',
	author='Wilson Estecio Marcilio Junior',
	author_email='wilson_jr@outlook.com',
	url='https://github.com/wilsonjr/sadire',
	license='MIT',
	install_requires=['pyqtree', 'numpy'],
	setup_requires=['pytest-runner'],
	tests_require=['pytest'],
	test_suite='tests',
)