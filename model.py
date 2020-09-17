from pytorch_lightning import LightningModule,Trainer,TrainResult,seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from torch_geometric.nn import SAGEConv 
from torch_geometric.data import Data,DataLoader
from torch.nn.functional import gelu,log_softmax,nll_loss
from torch.nn import Linear,BCEWithLogitsLoss
from torch import mean,max,stack
from torch.optim import Adam

from data import Load_Dataset

class GNN(LightningModule):
    def __init__(self,B_s,Node_Dim,Hidden_Dim,Out_Dim,Class_Dim,Loss_Only=True):
        super(GNN,self).__init__()
        self.loss_only = Loss_Only
        self.Bs = B_s
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
        self.Trainset,self.TestSet = Load_Dataset("PPI")
    def val_dataloader(self):
        ValLoader = DataLoader(self.TestSet,batch_size=self.Bs,shuffle=False)
        return ValLoader
    def train_dataloader(self):
        TrainLoader = DataLoader(self.Trainset,batch_size=self.Bs,shuffle=True)
        return TrainLoader
    def configure_optimizers(self):
        return Adam(self.parameters(),lr=1e-3)
    def Loss(self,logits,Y):
        return BCEWithLogitsLoss()(logits,Y)
    def training_step(self,Batch,Batch_Idx):
        Data = Batch
        logits = self(Data)
        loss = self.Loss(logits,Data.y)
        train_loss = {'train_loss':loss}
        if self.loss_only:
            Result = {"loss":loss,"log":train_loss}
        else:
            Acc_Bool = logits == Data.y
            Acc = sum(Acc_Bool.long()) * 100// len(logits)
            Result = {"loss":train_loss,"training_accuracy":Acc.float()}
        return Result
    def training_epoch_end(self,Outputs):
        Avg_Loss = stack([x['loss'] for x in Outputs]).mean()
        if self.loss_only:
            Epoch_Log = {"avg_training_loss":Avg_Loss}
            self.logger.experiment.log_metrics(Epoch_Log)
            return Epoch_Log
        else:
            Avg_Acc = stack([x['training_accuracy'] for x in Outputs]).mean()
            Epoch_Log = {"avg_training_loss":Avg_Loss,"avg_training_accuracy":Avg_Acc}
            self.logger.experiment.log_metrics(Epoch_Log)
            return Epoch_Log
    def validation_step(self,Batch,Batch_Idx):
        Val_Data = Batch
        Logits = self(Val_Data)
        loss = self.Loss(Logits,Val_Data.y)
        if self.loss_only:
            Result = {'val_loss':loss}
            self.logger.experiment.log_metrics(Result)
            return Result
        else:
            Acc_Bool = Logits == Val_Data.y
            Acc = sum(Acc_Bool.long()) * 100// len(Logits)
            Result = {"val_loss":loss,"val_accuracy":Acc.float()}
            self.logger.experiment.log_metrics(Result)
            return Result
        def valid_epoch_end(self,Outputs):
            Avg_Loss = stack([x['val_loss'] for x in Outputs]).mean()
            if self.loss_only:
                Epoch_Log = {"avg_val_loss":Avg_Loss}
                self.logger.experiment.log_metrics(Epoch_Log)
                return Epoch_Log
            else:
                Avg_Acc = stack([x['val_accuracy'] for x in Outputs]).mean()
                Epoch_Log = {"avg_val_loss":Avg_Loss,"avg_val_accuracy":Avg_Acc}
                self.logger.experiment.log_metrics(Epoch_Log)
                return Epoch_Log
if __name__ == "__main__":
    seed_everything(42)
    Logger = CSVLogger("logs",name="Trial",version="SAGEConv")
    Logger.save()
    Mod = GNN(2,50,150,200,121)
    trainer = Trainer(logger=Logger,max_epochs=1)
    trainer.fit(Mod)
    
