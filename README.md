# MasterV26
Code used and developed for my master's thesis, spring semster of 2026. The codebase includes code for preprocesing, labeling, extracting features from, analyzing and training ML models on a dataset consisting of IMU and pressure sensing insoles movement data. Some code is derived either from the previous work of Maria Ulseth Sylte (https://github.com/mariausy/Master_25) and Alba Casanica (https://github.com/AlbaCasanica/CT), as well as example code from my supervisor Roya Doshmanziari. In the files in which this is the case, credit as well as descriptions of modifications to the original code is given to the best of my ability in comments at the top of the files.

## Getting started with the codebase
Running the progrm requires a working installation of Python 3 as well as the Python packages listed in requirements.txt. Intall these (for instance in a virual environment):
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
