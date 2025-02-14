Make sure you have poetry installed
```pip install poetry```

Then, clone this repository
```
git clone https://github.com/Thomas2710/effective_complexity.git
cd effective_complexity
```

Install dependencies with poetry
```
poetry install
```
Activate the virtual environment
```
poetry shell
```
Then install the package with 
```
pip install -e .
```

Enter the main folder of the project
```
cd effective_complexity
```

Run the experiment with 
```
ec run -m __model_name__ -d __dataset_name__
```

or, with default settings:
```
ec run
```
