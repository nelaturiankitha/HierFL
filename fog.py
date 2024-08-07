import copy
from average import average_weights

class Fog():

    def __init__(self, id, eids, shared_layers):
        """
        id: fog id
        eids: ids of the edge servers under this fog node
        receiver_buffer: buffer for the received updates from selected edge servers
        shared_state_dict: state dict for shared network
        id_registration: participated edge servers in this round of aggregation
        sample_registration: number of samples of the participated edge servers in this round of aggregation
        all_sample_num: the total samples for all the edge servers under this fog node
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the fog node
        :param eids: Indexes of all the edge servers under this fog node
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.eids = eids
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_sample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.clock = []

    def refresh_fogserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = sum(edge.sample_registration.values())
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        self.receiver_buffer[edge_id] = eshared_state_dict
        return None

    def aggregate(self, args):
        """
        Aggregating updates from edge servers
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict,
                                                 s_num=sample_num)

    def send_to_edge(self, edge):
        edge.receive_from_fogserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_fog(fog_id=self.id,
                               fshared_state_dict=copy.deepcopy(
                                   self.shared_state_dict))
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None