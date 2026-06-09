# MasterV26
Code used and developed for my master's thesis, spring semster of 2026. The codebase includes code for preprocesing, labeling, extracting features from, analyzing and training ML models on a dataset consisting of IMU and pressure sensing insoles movement data. Data files are not provided here. Some code is derived either from the previous work of Maria Ulseth Sylte (https://github.com/mariausy/Master_25) and Alba Casanica (https://github.com/AlbaCasanica/CT), as well as example code from my supervisor Roya Doshmanziari. This is noted in comments at the top of the files for the relevant files.

Parts of the code were developed with assistance from ChatGPT-5. In most cases, AI assistance was limited to implementation support, debugging and plotting. In one instance, the definition of the transient score, chatGPT also contributed to the metric development. This is explicitly stated in the corresponding file. 

## Getting started with the codebase
Running the program requires a working installation of Python 3 as well as the Python packages listed in requirements.txt. Intall these (for instance in a virtual environment):
At the base directory (MasterV26) create a virtual environment (for instance named .venv) from terminal.
```bash
python -m venv .venv 
```
Activate the virtual environment using
```bash
.venv\Scripts\Activate
```
Install requirements
```bash
pip install -r requirements.txt
```
Run any .py file from the base directory using the terminal command
```bash
python -m folder_where_the_file_resides.name_of_file (not .py at the end of filename)
```
The labeling and preprocessing of files has been set up in a Jupyter notebook, named labeling_pipeline.ipynb. To run through this, follow instructions in the notebook. Set jupyter kernel to be .venv to have access to all required packages.

## Notes
This codebase is a work in progress, and parts of it may be somewhat messy or inconsistent.  
Where assumptions are made or implementation details may seem counterintuitive, they are generally explained in the code comments.

That said, the project is not perfect, and there may still be areas that could benefit from refactoring or clearer structure.
