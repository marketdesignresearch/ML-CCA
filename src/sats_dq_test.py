from pysats import PySats


gsvm = PySats.getInstance().create_gsvm(seed=10, isLegacyGSVM=False)
p = [0.0] * len(gsvm.get_good_ids())


for bidder_id in gsvm.get_bidder_ids():
    print(bidder_id)
    demanded_bundles = gsvm.get_best_bundles(bidder_id, p, 1)
    print(demanded_bundles)
