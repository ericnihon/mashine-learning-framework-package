from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

src_dir = '.'

setup(
    name='uenn',
    version='1.0.0',
    url='https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-material/ws20/team-16/examsubmission',
    license='',
    long_description=long_description,
    author='Uli Binder, Eric Dienhart',
    author_email='uli.binder@stud-mail.uni-wuerzburg.de, eric.dienhart@stud-mail.uni-wuerzburg.de',
    description='Machine Learning Framework',
    install_requires=['numpy', 'pandas', 'os', 'fnmatch', 'seaborn', 'matplotlib', 'tqdm'],
    package_dir={'': src_dir},
    packages=find_packages(where=src_dir),
)
