
import torch
import torch.nn as nn

cfg = {
    'VGG11_CIFAR': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512],
    'VGG13_CIFAR': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512],
    'VGG16_CIFAR': [64, 64, 'M', 128, 128, 'M',
                    256, 256, 256, 'M', 512, 512, 512],
    'VGG19_CIFAR': [64, 64, 'M', 128, 128, 'M',
                    256, 256, 256, 256, 'M', 512, 512, 512, 512],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels=3, num_classes=10, image_size=32):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], in_channels)
        # Dynamically compute the input features for the classifier
        self.classifier_input_features = self._get_classifier_input_features(in_channels, image_size)
        print(f"Classifier input features: {self.classifier_input_features}")

        self.classifier = nn.Linear(self.classifier_input_features, num_classes)

    def forward(self, x):
        print(f"Input x shape: {x.shape}")
        out = self.features(x)
        print(f"Output of features shape: {out.shape}")
        out = out.view(out.size(0), -1)
        print(f"Flattened output shape: {out.shape}")
        out = self.classifier(out)
        return out


    def _make_layers(self, cfg_list, in_channels):
        layers = []
        for x in cfg_list:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)

    def _get_classifier_input_features(self, in_channels, image_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, image_size, image_size)
            output = self.features(dummy_input)
            num_features = output.view(1, -1).size(1)
        return num_features

def test():
    net = VGG('VGG16_CIFAR', in_channels=3, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print('Output shape:', y.size())

# Uncomment to run the test
test()
