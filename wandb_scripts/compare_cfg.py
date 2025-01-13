import wandb

api = wandb.Api()

run1 = api.run("farai/lp-cleanba/runs/tvqm1z59")
run2 = api.run("farai/lp-cleanba/runs/bkynosqi")

cfg1 = run1.config
cfg2 = run2.config

def recursive_diff(cfg1, cfg2, prefix=""):
    if cfg1 == cfg2:
        return {}
    if not isinstance(cfg1, dict) or not isinstance(cfg2, dict):
        return {prefix.rstrip("."): (cfg1, cfg2)}
    diff = {}
    for k, v in cfg1.items():
        if v != cfg2.get(k, None):
            diff.update(recursive_diff(v, cfg2.get(k, None), prefix=f"{prefix}{k}."))

    for k, v in cfg2.items():
        if k not in cfg1:
            diff[f"{prefix}{k}"] = (None, v)

    return diff

diff = recursive_diff(cfg1, cfg2)
for k, v in diff.items():
    print(k, v)