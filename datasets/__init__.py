from .sunrefer import SUNREFER, SUNREFER_PREDET
from .scanrefer import SCANREFER

def create_dataset(args, split):
    if args.dataset == "sunrefer":
        dataset = SUNREFER(args, split)
    elif args.dataset == "sunrefer_predet":
        dataset = SUNREFER_PREDET(args, split)
    elif args.dataset == "scanrefer":
        dataset = SCANREFER(args, split)
    else:
        raise ValueError("Wrong dataset")
    return dataset