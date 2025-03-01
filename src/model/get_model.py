from src.model.basic_modules import Generator, Discriminator

def get_model(args):
    if args.data_name == '256':
        img_res = 256
    elif args.data_name == '512':
        img_res = 512
    G = Generator(z_dim=img_res, c_dim=0, w_dim=img_res, img_resolution=img_res, img_channels=3, use_edge = True, use_line = False, use_ln = True, few_high = False)
    D = Discriminator(c_dim=0, img_resolution=img_res, img_channels=3)

    print(f'Generator: {count_parameters(G)}')
    print(f'Discriminator: {count_parameters(D)}')
    return G, D

def count_parameters(model, debug=False):
    params = [
        (name, p.numel()) for name, p in model.named_parameters() if p.requires_grad
    ]
    
    # 파라미터 개수 기준으로 내림차순 정렬
    params_sorted = sorted(params, key=lambda x: x[1], reverse=True)
    
    total_params = sum(p[1] for p in params_sorted)
    
    if debug:
        for name, num_params in params_sorted:
            print(f"{name}: {num_params} parameters")
    
    print(f"Total Trainable Parameters: {total_params}, {total_params/1e6:.2f}M")
    return total_params

if __name__ == '__main__':
    from argparse import Namespace
    args = Namespace(data_name='256')
    get_model(args)