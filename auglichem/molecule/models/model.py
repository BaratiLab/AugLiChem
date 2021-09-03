def get_model(task_name, config):
    if config['type'] == 'gin':
        from auglichem.models.gine import GINE
        return GINE(task_name, **config)
    elif config['type'] == 'gcn':
        from auglichem.models.gcn import GCN
        return GCN(task_name, **config)
    elif config['type'] == 'afp':
        from auglichem.models.attentive_fp import AttentiveFP
        return AttentiveFP(task_name, **config)
