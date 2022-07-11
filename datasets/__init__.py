from .sunrefer import SUNREFER

def create_dataset(args, split):
    if args.dataset == "sunrefer":
        dataset = SUNREFER(args, split)
    else:
        raise ValueError("Wrong dataset")
    return dataset