from multiprocessing import freeze_support
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import inchi
import sys
from rdkit.Chem.rdmolops import *
from src.config import data_path
import os
import pandas as pd
import numpy as np

def sdf_to_desc(file_name):
    freeze_support()

    sdf = Chem.SDMolSupplier(os.path.join(data_path, file_name + ".sdf"))
    mols = []

    for mol in sdf:
        if mol:
            mols.append(mol)

    # Create Calculator
    calc = Calculator(descriptors)

    # map method calculate multiple molecules (return generator)
    print(list(calc.map(mols)))

    # pandas method calculate multiple molecules (return pandas DataFrame)
    #print(calc.pandas(mols))

    # save data frame
    calc.pandas(mols).to_csv(os.path.join(data_path, "mordred_files", file_name + ".csv"))

def sdf_to_fp(filename):
    molecules = Chem.SDMolSupplier(os.path.join(data_path, file_name + ".sdf"))
    mols=[]
    for mol in molecules:
        if mol:
            #fp = RDKFingerprint(mol, fpSize=2048)
            fp = RDKFingerprint(mol, fpSize=2048, minPath=1, maxPath=7, nBitsPerHash=2, useHs=1)
            mols.append([fp.ToBase64(),str(mol.GetProp("_Name"))])
            #print(str(fp.ToBase64()) + "\t" + str(mol.GetProp("_Name")))
    np.asarray(mols)
    pd.DataFrame(mols).to_csv(os.path.join(data_path, "fp", file_name + ".csv"))


def sdf_to_inchikey():
    molecules = Chem.SDMolSupplier(sys.argv[1])

    csv = open(sys.argv[1] + ".inchikey", "w")

    for mol in molecules:
        if mol:
            csv.write(inchi.MolToInchiKey(mol) + " " + mol.GetProp("_Name") + "\n")

    csv.close()


if __name__ == '__main__':
    sdf_to_desc("compound_set1")
    sdf_to_desc("compound_set2")
    sdf_to_desc("compound_set3")