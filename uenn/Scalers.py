class NormalScaler:
    def fit(self, data):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0) - self.min

    def transform(self, data):
        return (data - self.min) / self.max

    def inverse_transform(self, scaled_data):
        return scaled_data * self.max + self.min


class StandardScaler:
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std == 0] = 0.00001

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, scaled_data):
        return scaled_data * self.std + self.mean
