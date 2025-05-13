from utils.prediction import predict

predict(
    './suppl/LigPose_param.chk',                                        # path to LigPose parameters
    device='cuda:0',                                                    # device
    protein='example/example_files/4r6e/4r6e_protein.pdb',              # path to protein (receptor) file
    ligand='example/example_files/4r6e/4r6e_ligand.mol2',               # path to ligand file, or SMILES
    ref_pocket_center='example/example_files/4r6e/4r6e_ligand.mol2',    # a file for selecting pocket atoms (e.g. predictions from Fpocket)
    cache_path='./cache',                                               # a temporary path for saving processed files
    pred_type=['structure'],                                            # tasks (['structure'] for structure prediction // ['screening'] for virtual screening // ['structure', 'screening'])
    output_structure_path='./output_structures',                        # path to saving output structures
    output_result_path='./output.csv',                                  # path to saving output records (in csv format)
)