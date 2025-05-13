# LigPose

Code related to: One-step Structure Prediction and Screening for Protein-Ligand Complexes using Geometric Deep Learning

LigPose is a docking-free geometric deep learning method to model the native-like conformation of ligands 
with their corresponding binding strengths. Specifically, for a given protein and ligand pair, 
the ligand and its binding target are jointly represented as a graph. 
Then, their 3-D structures are directly optimized by the atom coordinates in the Euclidean space, 
with the binding strength and the correlation-enhanced graph learning jointly learned as auxiliary tasks.


<details open><summary><b>Table of contents</b></summary>

- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
</details>


## Installation <a name="installation"></a>

LigPose is implemented in PyTorch (tested on Python 3.6 in Linux system). All basic Python dependencies are listed in `requirements.txt`. 
Run the following command in your conda virtual environment (expected to take a few minutes):
  ```sh
  pip install -r requirements.txt
  ```

## Usage <a name="usage"></a>

### Inference <a name="inference"></a>


After installing all dependencies, you can use the LigPose as follows at the root path of this repository (shown in `demo.py`)
(expected to take a few seconds with a single GPU device). 
Model parameters can be found [here](https://1drv.ms/u/c/469b767efa9cca5a/EYKauV0h1dZHhZuNKcWzb9wBZFpOheesP8wnSdt_ImuZqw?e=R3kJ22).
```python
from utils.prediction import predict

predict(
    './LigPose_param.chk',                                              # path to LigPose parameters
    device='cuda:0',                                                    # device
    protein='example/example_files/4r6e/4r6e_protein.pdb',              # path to protein (receptor) file
    ligand='example/example_files/4r6e/4r6e_ligand.mol2',               # path to ligand file, or SMILES
    ref_pocket_center='example/example_files/4r6e/4r6e_ligand.mol2',    # a file for selecting pocket atoms (e.g. predictions from Fpocket)
    cache_path='./cache',                                               # a temporary path for saving processed files
    pred_type=['structure'],                                            # tasks (['structure'] for structure prediction // ['screening'] for virtual screening // ['structure', 'screening'])
    output_structure_path='./output_structures',                        # path to saving output structures
    output_result_path='./output.csv',                                  # path to saving output records (in csv format)
)
```

We also show an example in `exampled/batch_prediction.ipynb` to perform batch prediction.


### Training <a name="training"></a>

To train with pdbbind data, first you can run the following command to prepare input data:
```bash
python preprocess/prepare_pdbbind.py --data_path path/to/pdbbind --data_suppl_path path/to/INDEX_general_PL_data.year 
--output_path path/to/output --cache ./cache
```
NOTE: RDKit can not handle some ligand.mol2 files, we recommend to use Open Babel to convert these molecules into 
pdb (XXXX_ligand.pdb) and smi (XXXX_ligand.smi) files and place them in the original folder.

The following command is an example for training a toy LigPose (small model size).
```bash
python -u train/train_pdbbind_struct.py \
--pdbbind_path path/to/prepared/data \
--original_path path/to/pdbbind \
--data_list_path eval/pdbbind/core_test \
--core_list_path eval/pdbbind/core_list.txt \
--batch_size 4 \
--lr 0.0001 \
--n_epoch 200 \
--max_len_after_sampling 400 \
--max_len_after_sampling_for_eval 400 \
--n_cycle 4 \
--node_hidden 64 \
--edge_hidden 32 \
--n_block 4 \
--cache_path ./cache \
--gpu_list 0
```

### License <a name="license"></a>

### Contact <a name="contact"></a>

### Acknowledgements <a name="acknowledgements"></a>

















