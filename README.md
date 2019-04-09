# Code framework for KG completion.
- Code:
    - modify based on [knowledge_representation_pytorch](https://github.com/jimmywangheng/knowledge_representation_pytorch)
- Dataset: this dataset comes from Know-Evolve repo.
    - 1st column in train.txt - subject entity
    - 2nd column - relation
    - 3rd column - object entity
    - 4th column - time

    - 1st figure in stat.txt - number of entities
    - 2nd figure in stat.txt - number of relations
    
    used `timestamp2datetime.py` and `preprocess.py` to make data for TATransE and TADistMult.  
    used `preprocess_TTransE.py` to make data for TTransE.  

- data.py: this is for corrupting triples and other functions for data

- util.py: this is collection of frequent functions

- evaluation.py: evaluation codes

- TTransE.py, TATransE.py, TADistMult.py: train codes

- You can run the code with
	```
	python TTransE.py (-- parameters)
	python TATransE.py
	python TADistMult.py
	```

