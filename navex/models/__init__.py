

def get_vgg():
    from .vgg import VGG
    return VGG


MODELS = {
    'own_vgg': get_vgg,
    'own_mobilenet': None,   # TODO
    'own_resnet': None,      # TODO
}
