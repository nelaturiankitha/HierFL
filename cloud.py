import copy
from average import average_weights

class Cloud():

    def __init__(self, shared_layers):
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.shared_state_dict = shared_layers.state_dict()
        self.clock = []

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def fog_register(self, fog):
        self.id_registration.append(fog.id)
        self.sample_registration[fog.id] = sum(fog.sample_registration.values())
        return None

    def receive_from_fog(self, fog_id, fshared_state_dict):
        self.receiver_buffer[fog_id] = fshared_state_dict
        return None

    def aggregate(self, args):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict,
                                                 s_num=sample_num)
        return None

    def send_to_fog(self, fog):
        fog.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None

    # Keep these methods for potential direct communication with edge servers
    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        self.receiver_buffer[edge_id] = eshared_state_dict
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None