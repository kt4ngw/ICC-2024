from args import input_options
from src.trainers.fedavg import FedAvgTrainer
from src.trainers.propose import ProposeTrainer
from src.trainers.propose2 import Propose2Trainer





def main():
    trainer= input_options()
    trainer.train()
if __name__ == '__main__':
    main()

