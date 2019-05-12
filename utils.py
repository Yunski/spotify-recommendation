from models import MF, GMF, MLP, NeuMF, ConvNCF

def load_model(name):
    models = {
        'mf': MF,
        'gmf': GMF,
        'mlp': MLP,
        'neumf': NeuMF,
        'convncf': ConvNCF
    }
    return models[name]
