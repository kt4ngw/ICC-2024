from src.trainers.base import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
from src.algorithms.algorithm2 import Algorithm2
class Propose2Trainer(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power, g_N0):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        # self.optimizer = MyAdam(model.parameters(), lr=options['lr'])
        self.optimizer = GD(model.parameters(), lr=options['lr'])
        super(Propose2Trainer, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, g_N0, model
                                             , self.optimizer)
        self.prob = Algorithm2(self.cost.localE, self.cost.uploadE, self.cost.localD, self.cost.uploadD, clients_label,
                              dataset).getProba()  # 输入带宽

    def train(self):
        print('>>> Select {} clients per round \n'.format(int(self.per_round_c_fraction * self.clients_num)))

        # Fetch latest flat model parameter
        self.latest_global_model = self.get_model_parameters()
        for round_i in range(self.num_round):
            # print("{}, {}".format(round_i, self.latest_global_model))

            # Test latest model on train data
            # self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_testdata(round_i)

            # Choose K clients prop to data size
            selected_clients = self.select_clients_with_prob()

            # Solve minimization locally
            local_model_paras_set, stats = self.local_train(round_i, selected_clients)

            self.metrics.update_cost(round_i, self.cost.delay_Sum, self.cost.energy_Sum)


            # Track communication cost
            self.metrics.extend_communication_stats(round_i, stats)

            # Update latest model
            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)
            self.optimizer.inverse_prop_decay_learning_rate(round_i)
           # self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train data
        # self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_testdata(self.num_round)

        # # Save tracked information
        self.metrics.write()



    def select_clients_with_prob(self, ):
        num_clients = min(int(self.per_round_c_fraction * self.clients_num), self.clients_num)
        index = np.random.choice(len(self.clients), num_clients, replace=False, p=self.prob)
        select_clients = []
        for i in index:
            select_clients.append(self.clients[i])
        return select_clients
