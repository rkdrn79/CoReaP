from src.model.basic_modules import Generator, Discriminator

def get_model(args):
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3, use_4input = True, use_ln = True)
    D = Discriminator(c_dim=0, img_resolution=512, img_channels=3)

    print(f'Generator: {count_parameters(G)}')
    print(f'Discriminator: {count_parameters(D)}')
    return G, D

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)