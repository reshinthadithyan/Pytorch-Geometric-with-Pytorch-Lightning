from torch_geometric.datasets import Planetoid
import os.path as osp
def Load_Dataset(Name):
    path = osp.join(osp.realpath(__file__),'..','data','dataset')
    Dataset =Planetoid(path,Name)
    return Dataset