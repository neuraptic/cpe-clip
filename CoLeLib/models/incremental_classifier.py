from torch import nn


class IncrementalClassifier(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.classifier = nn.Linear(num_in, num_out)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.actual_classes = num_out

    def forward(self, x):
        return self.classifier(x)

    def adaptation(self, actual_classes):
        if actual_classes > self.actual_classes:
            old_w, old_b = self.classifier.weight.data, self.classifier.bias.data
            self.classifier = nn.Linear(self.classifier.in_features, actual_classes)
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
            self.classifier.weight.data[:self.actual_classes] = old_w
            self.classifier.bias.data[:self.actual_classes] = old_b
            self.actual_classes = actual_classes
        else:
            return
