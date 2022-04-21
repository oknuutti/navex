from setuptools import setup, find_packages


setup(
    name='navex',
    version='1.0',
#    package_dir={'navex': ''},
    packages=find_packages(include=['navex*']),
    include_package_data=True,
    package_data={'navex.experiments': ['*.yaml']},

    # Declare your packages' dependencies here, for eg:
    install_requires=['tqdm', 'pillow', 'numpy', 'scipy', 'matplotlib',  # from conda regular channel
                      'r2d2 @ git+https://github.com/oknuutti/r2d2',
                      # 'pytorch-lightning' from conda-forge
                      # 'pytorch', 'torchvision', 'cudatoolkit',  # from conda pytorch channel, don't seem to work with pip
                      ],

    author='Olli Knuuttila',
    author_email='olli.knuuttila@gmail.com',

    description='CNN-based feature extractor for visual navigation and SLAM near asteroids',
    url='https://github.com/oknuutti/navex',
    license='MIT',
)
