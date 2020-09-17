from torch_geometric.datasets import PPI
import os.path as osp
def Load_Dataset(Name):
    path = osp.join(osp.realpath(__file__),'..','data','PPIdataset')
    if Name == "PPI":
        Train_Dataset = PPI(path,split="train")
        Valid_Dataset = PPI(path,split="val") 
        return Train_Dataset,Valid_Dataset
if __name__ == '__main__':
    A,_ = Load_Dataset("PPI")
    print(A[0])