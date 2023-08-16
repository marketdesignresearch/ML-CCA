from collections import OrderedDict


class ValueModel:
    def __init__(self, name, number_of_items, local_bidder_ids, regional_bidder_ids, national_bidder_ids,
                 highFrequency_bidder_ids, scaler):
        self.name = name
        self.local_bidder_ids = local_bidder_ids  # mrvm & srvm
        self.regional_bidder_ids = regional_bidder_ids
        self.national_bidder_ids = national_bidder_ids
        self.highFrequency_bidder_ids = highFrequency_bidder_ids  # only srvm
        self.scaler = scaler
        self.M = number_of_items
        self.N = len(local_bidder_ids) + len(regional_bidder_ids) + len(national_bidder_ids) + len(
            highFrequency_bidder_ids)
        self.scaler = scaler

    def _trafo(self, NN_parameter):
        parameters = OrderedDict()
        for key, value in NN_parameter.items():
            if key.lower() == 'Local'.lower():
                bidders = self.local_bidder_ids
            if key.lower() == 'High_Frequency'.lower():
                bidders = self.highFrequency_bidder_ids
            if key.lower() == 'Regional'.lower():
                bidders = self.regional_bidder_ids
            if key.lower() == 'National'.lower():
                bidders = self.national_bidder_ids

            for bidder_id in bidders:
                parameters['Bidder_{}'.format(bidder_id)] = value
        return (parameters)

    def parameters_to_bidder_id(self, NN_parameters):
        return self._trafo(NN_parameters)
