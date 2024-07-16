def clean_args(args):
    if hasattr(args, 'col_stats'):
        args.col_stats = str(args.col_stats) if args.col_stats else None
    if hasattr(args, 'col_names_dict'):
        args.col_names_dict = str(args.col_names_dict) if args.col_names_dict else None
    return args
