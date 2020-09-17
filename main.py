from pytorch_lightning import LightningModule,Trainer,TrainResult,seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger

from data import Load_Dataset
from model import GNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("Batch_Size",type=int)
parser.add_argument("Max_Epoch", type=int)
Args = parser.parse_args()


Epochs_Max = Args.Max_Epoch
Batch_Size = Args.Batch_Size

seed_everything(42)
Logger = CSVLogger("logs",name="Trial_SAGEConv",version=str(Batch_Size)+"_"+str(Epochs_Max))
Trainset,TestSet = Load_Dataset("PPI")
Mod = GNN(Batch_Size,50,500,200,121)
trainer = Trainer(logger=Logger,max_epochs=Epochs_Max)
trainer.fit(Mod)