## DeiT III Environment Setup

It uses [NVIDIA apex](https://github.com/NVIDIA/apex). Thus, you must compile apex. Begin by creating a new virtual environment for Python, we use conda and Python 3.10. Then: 

Clone apex:
```bash
git clone https://github.com/NVIDIA/apex.git
```

Run:
```bash
cd apex
git checkout 2386a912164
python setup.py install --cuda_ext --cpp_ext
```

Now that you have compiled apex, you can continue setting up the Python environment. Either do so by installing the packages in the `deit/requirements.txt` manually or by inheriting our conda environment by running:
```bash
conda env update --file deit/environment.yml
```