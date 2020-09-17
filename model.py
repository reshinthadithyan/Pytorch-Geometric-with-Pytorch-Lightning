from pytorch_lightning import LightningModule,Trainer
from torch_geometric.nn import SAGEConv 
from torch_geometric.data import Data  
from torch.nn.functional import gelu,log_softmax
from torch.nn import Linear

from data import Load_Dataset

class GNN(LightningModule):
    def __init__(self,Node_Dim,Hidden_Dim,Out_Dim,Class_Dim):
        super(GNN,self).__init__()
        self.Data = None
        self.Conv1 = SAGEConv(Node_Dim,Hidden_Dim)
        self.Conv2 = SAGEConv(Hidden_Dim,Out_Dim)
        self.ClassHead = Linear(Out_Dim,Class_Dim)  
    def forward(self,Data):
        '''Simple SAGE Pass'''
        X,Edge_Index = Data.x,Data.edge_index
        X = gelu(self.Conv1(X,Edge_Index))
        X  = gelu(self.Conv2(X,Edge_Index))
        X = log_softmax(self.ClassHead(X),dim=1)
        return X
if __name__ == "__main__":
    Dataset = Load_Dataset("Cora")
    print(Dataset[0].x[0])
    Mod = GNN(1433,500,200,7)
    print(Mod(Dataset[0]).size())