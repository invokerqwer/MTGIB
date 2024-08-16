import pandas as pd
import torch
from rdkit.Chem.Draw import SimilarityMaps
import random
import numpy as np
from utils import create_batch_mask
import os
import argument
import time
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from torch_geometric.data import DataLoader
from utils import get_stats, write_summary, write_summary_total
IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 600,600
torch.set_num_threads(2)
df = pd.read_csv(f'data/raw_data/ZhangDDI_train.csv', sep=",")


def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
args, unknown = argument.parse_args()
    
print("Loading dataset...")
start = time.time()

# Load dataset
train_set = torch.load("./data/processed/{}_train.pt".format(args.dataset))

print("Dataset Loaded! ({:.4f} sec)".format(time.time() - start))
from models.DISE_cont import DISE
device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
model = DISE(device = device, tau = args.tau, num_step_message_passing = args.message_passing,EM=args.EM_NUM).to(device)
PATH='state_dict_model0.0001,0.0001,{}.pth'.format(args.EM_NUM)
model.load_state_dict(torch.load(PATH))
train_loader = DataLoader(train_set, batch_size = 1, shuffle=False)
i=0
z=0
ls = [193]
for bc, samples in enumerate(train_loader):
    if i==ls[z]:
        z=z+1
        masks = create_batch_mask(samples)
        
        solute_sublist,solvent_sublist = model.get_subgraph([samples[0].to(device), samples[1].to(device), masks[0].to(device), masks[1].to(device)],bottleneck=True)
        os.makedirs("solute/"+str(i)+"/"+df.iloc[i]["smiles_1"])
        mol = Chem.MolFromSmiles(df.iloc[i]["smiles_1"])
        solu_atom_num=mol.GetNumAtoms()
        mol.RemoveAllConformers()
        print(solute_sublist)
        for j in range(len(solute_sublist)):
            if j!=139:
                solvent_fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,solute_sublist[j].cpu().detach().numpy() ,colorMap='RdBu',
                                                                alpha=0.05,
                            size=(200,200))
                solvent_fig.savefig("solute/"+str(i)+"/"+df.iloc[i]["smiles_1"]+"/solute{}.png".format(j), bbox_inches='tight', dpi=600)
        os.makedirs("solvent/"+str(i)+"/"+df.iloc[i]["smiles_2"])
        mol = Chem.MolFromSmiles(df.iloc[i]["smiles_2"])
        solu_atom_num=mol.GetNumAtoms()
        mol.RemoveAllConformers()
        print(solvent_sublist)
        for j in range(len(solvent_sublist)):
            if j!=139:
                solvent_fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,solvent_sublist[j].cpu().detach().numpy() ,colorMap='RdBu',
                                                                alpha=0.05,
                            size=(200,200))
                solvent_fig.savefig("solvent/"+str(i)+"/"+df.iloc[i]["smiles_2"]+"/solute{}.png".format(j), bbox_inches='tight', dpi=600)
    i = i+1
#print(1)
