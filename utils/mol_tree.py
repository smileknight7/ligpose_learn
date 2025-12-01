
import pandas as pd
import rdkit.Chem as Chem
from tqdm.auto import tqdm
import numpy as np
from utils.chemutils import tree_decomp, get_clique_mol
import matplotlib.pyplot as plt
from rdkit import Chem
# from rdkit.Chem import Draw

class MolTreeNode(object):

    def __init__(self, cmol):
        self.smiles = Chem.MolToSmiles(cmol, canonical=True)
        self.mol = cmol


class MolTree_process(object):
    def __init__(self, mol):
        self.smiles = Chem.MolToSmiles(mol)
        self.mol = mol
        self.num_rotatable_bond = 0
        '''
        # use reference_vocab and threshold to control the size of vocab
        reference_vocab = np.load('./utils/reference.npy', allow_pickle=True).item()
        reference = defaultdict(int)
        for k, v in reference_vocab.items():
            reference[k] = v'''

        # use vanilla tree decomposition for simplicity
        cliques, edges = tree_decomp(self.mol, reference_vocab=None)
        self.nodes = []
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(self.mol, cmol, c)
            self.nodes.append(node)
            

if __name__ == "__main__":

    MolTree_process()


