# evaluation/evaluate_gtrs.py
def evaluate(model, dataset):
    bins = {
        "high": [],
        "medium": [],
        "all": []
    }

    for s in dataset:
        pred = model(s["image"].cuda()).cpu()
        gt = s["depth"]

        m = {
            "AbsRel": absrel(pred, gt),
            "RMSE": rmse(pred, gt),
            "δ1": delta(pred, gt, 1.25)
        }

        bins["all"].append(m)
        if s["gtrs"] > 0.9:
            bins["high"].append(m)
        elif s["gtrs"] > 0.75:
            bins["medium"].append(m)

    return bins
