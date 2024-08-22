


class Cost(object):
    def __init__(self, clients, ):
        self.clients = clients
        self.energy_Sum = 0
        self.delay_Sum = 0
        self.localE = []
        self.uploadE = []
        self.localD = []
        self.uploadD = []
        self.update_para_cost()
    def update_para_cost(self):
        for client in self.clients:
            self.localE.append(client.getLocalEngery())
            self.uploadE.append(client.getUploadEngery())
            self.localD.append(client.getLocalDelay())
            self.uploadD.append(client.getUploadDelay())


class ClientAttr(object):
    def __init__(self, cpu_frequency, B, transmit_power, g_N0, ):
        self.cpu_frequency = cpu_frequency
        self.bandwidth = B
        self.transmit_power = transmit_power
        self.g_N0 = g_N0

    def get_client_attr(self, id):
        return {
            "cpu_frequency": self.cpu_frequency[id],
            "B": self.bandwidth[id],
            "transmit_power": self.transmit_power[id],
            "g_N0": self.g_N0[id]
        }
