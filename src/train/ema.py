import torch


class ExponentialMovingAverage:
    def __init__(self, parameters, decay, use_num_updates=True):
        if not (0.0 <= decay <= 1.0):
            raise ValueError("Decay must be between 0 and 1.")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates)) if self.num_updates is not None else self.decay
        if self.num_updates is not None:
            self.num_updates += 1
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow_params': self.shadow_params
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']
