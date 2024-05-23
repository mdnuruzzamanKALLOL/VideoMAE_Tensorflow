import json
import tensorflow as tf

def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1

class LayerDecayValueAssigner:
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))

def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for var in model.trainable_variables:
        var_name = var.name
        if not var.trainable:
            continue  # frozen weights
        if len(var.shape) == 1 or var_name.endswith(".bias") or var_name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(var_name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(var)
        parameter_group_names[group_name]["params"].append(var_name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = set()
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.trainable_variables

    opt_args = dict(learning_rate=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['epsilon'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['beta_1'], opt_args['beta_2'] = args.opt_betas

    print("optimizer settings:", opt_args)

    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('epsilon', None)
        optimizer = tf.keras.optimizers.SGD(momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('epsilon', None)
        optimizer = tf.keras.optimizers.SGD(momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = tf.keras.optimizers.Adam(**opt_args)
    elif opt_lower == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(**opt_args)
    elif opt_lower == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(**opt_args)
    elif opt_lower == 'radam':
        optimizer = tf.keras.optimizers.experimental.RAdam(**opt_args)
    elif opt_lower == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(**opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(rho=0.9, momentum=args.momentum, **opt_args)
    else:
        raise ValueError(f"Invalid optimizer: {opt_lower}")

    return optimizer