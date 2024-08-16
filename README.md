# MTGL-ADMET: A Novel Multi-Task Graph Learning Framework for ADMET Prediction Enhanced by Status-Theory and Maximum Flow
A Multi-Task Graph Learning framework for predicting multiple ADMET properties of drug-like small molecules (MTGL-ADMET) under a new paradigm of MTL, "one primary, multiple auxiliaries".
# Step-by-step running:

## 1. Requirements
+ Python == 3.8
+ dgl == 0.4.3
+ scikit-learn == 0.24.2
+ pandas == 1.2.0
+ numpy == 1.20.2
+ rdkit 


## 2. Create data 
Obtain graph data

Running
```sh
python create_graph_data.py
```

## 3. Running

```sh
python Training.py
```
Here, we used ‘CYP2C9’ as the example. As for other ADMET endpoints, the model can be adjusted according to the auxiliary tasks.

# Acknowledgements
Part of the code was adopted from [1].

# References
[1] Wu Z, Jiang D, Wang J, et al. Mining toxicity information from large amounts of toxicity data[J]. Journal of Medicinal Chemistry, 2021, 64(10): 6924-6936.
