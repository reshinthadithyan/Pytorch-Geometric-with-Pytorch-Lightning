from pytorch_lightning import LightningModule,Trainer,TrainResult,seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger

from data import Load_Dataset
from model import GNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("Max_Epoch", type=int)
Args = parser.parse_args()


Epochs_Max = Args.Max_Epoch

seed_everything(42)
Logger = CSVLogger("logs",name="Trial_SAGEConv")
Dataset = Load_Dataset("Cora")
Mod = GNN(1433,500,200,7)
trainer = Trainer(logger=Logger,max_epochs=Epochs_Max)
trainer.fit(Mod)