from pytorch_lightning import LightningModule,Trainer,TrainResult,seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from torch_geometric.nn import SAGEConv 
from torch_geometric.data import Data  
from torch.nn.functional import gelu,log_softmax,nll_loss
from torch.nn import Linear
from torch import mean,max
from torch.optim import Adam

from data import Load_Dataset

class GNN(LightningModule):
    def __init__(self,Node_Dim,Hidden_Dim,Out_Dim,Class_Dim):
        super(GNN,self).__init__()
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
    def prepare_data(self):
        self.Data = Load_Dataset("Cora")
    def train_dataloader(self):
        return self.Data
    def configure_optimizers(self):
        return Adam(self.parameters(),lr=1e-3)
    def training_step(self,Batch,Batch_Idx):
        Data = Batch
        logits = self(Data)
        loss = nll_loss(logits,Data.y)
        Acc_Bool = max(logits,dim=1)[1] == Data.y
        Acc = sum(Acc_Bool.long()) * 100// len(logits)
        print(loss,Acc)
        Result = {"loss":loss,"training_accuracy":Acc}
        self.logger.experiment.log_metrics(Result)
        #result = TrainResult(minimize=loss)
        #result.log('train_loss',loss,on_epoch=False,prog_bar=False,logger=True,reduce_fx=mean)
        return Result
if __name__ == "__main__":
    seed_everything(42)
    Logger = CSVLogger("logs",name="Trial",version="SAGEConv")
    Logger.save()
    Dataset = Load_Dataset("Cora")
    Mod = GNN(1433,500,200,7)
    trainer = Trainer(logger=Logger,max_epochs=1)
    trainer.fit(Mod)
    
