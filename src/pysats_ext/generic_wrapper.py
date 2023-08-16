from pysats.simple_model import SimpleModel

from jnius import (
    JavaClass,
    MetaJavaClass,
    JavaMethod,
    JavaMultipleMethod,
    cast,
    autoclass,
)
from torch.distributed.autograd import context

Random = autoclass("java.util.Random")
HashSet = autoclass("java.util.HashSet")
LinkedList = autoclass("java.util.LinkedList")
HashSet = autoclass("java.util.HashSet")
Bundle = autoclass("org.marketdesignresearch.mechlib.core.Bundle")
BundleEntry = autoclass("org.marketdesignresearch.mechlib.core.BundleEntry")
LinkedHashMap = autoclass("java.util.LinkedHashMap")
Price = autoclass("org.marketdesignresearch.mechlib.core.price.Price")
LinearPrices = autoclass("org.marketdesignresearch.mechlib.core.price.LinearPrices")
SizeBasedUniqueRandomXOR = autoclass(
    "org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR"
)
JavaUtilRNGSupplier = autoclass(
    "org.spectrumauctions.sats.core.util.random.JavaUtilRNGSupplier"
)

class GenericWrapper(SimpleModel):
    def __init__(self, model):
        self._pysats_model = model
        self.population = model.population
        self.goods = {}
        self.goods_supply = []
        self.good_to_licence = {}
        self.licence_to_good = {}
        self.mip_path = model.mip_path
        self.efficient_allocation = None

        world = self.population[0].getWorld()
        self._bidder_list = model._bidder_list

        # Store bidders
        bidderator = self._bidder_list.iterator()
        count = 0
        while bidderator.hasNext():
            bidder = bidderator.next()
            self.population[count] = bidder
            count += 1

        # Store goods
        self._good_to_id = {}
        self.licence_to_good = [0 for g in self._pysats_model.get_good_ids()]
        goods_iterator = world.getAllGenericDefinitions().iterator()
        count = 0
        while goods_iterator.hasNext():
            good = goods_iterator.next()
            self.goods[count] = good
            self._good_to_id[good.getName()] = count
            self.goods_supply.append(good.getQuantity())
            # good to licence mapping
            contained_goods_iterator = good.containedGoods().iterator()
            licenceIds = []
            while contained_goods_iterator.hasNext():
                licence = contained_goods_iterator.next()
                licenceIds.append(licence.getLongId())
                self.licence_to_good[licence.getLongId()] = count
            self.good_to_licence[count] = licenceIds
            count += 1

        # licence to good mapping

    def get_best_bundles(
        self, bidder_id, price_vector, max_number_of_bundles, allow_negative=False
    ):
        assert len(price_vector) == len(self.goods.keys())
        bidder = self.population[bidder_id]
        prices_map = LinkedHashMap()
        index = 0
        for good in self.goods.values():
            prices_map.put(good, Price.of(price_vector[index]))
            index += 1
        bundles = bidder.getBestBundles(
            LinearPrices(prices_map), max_number_of_bundles, allow_negative
        )
        result = []
        for bundle in bundles:
            bundle_vector = []
            for i in range(len(price_vector)):
                bundle_vector.append(bundle.countGood(self.goods[i]))
            result.append(bundle_vector)
        return result

    def get_efficient_allocation(self, display_output=False):
        if self.efficient_allocation:
            return self.efficient_allocation, sum(
                [
                    self.efficient_allocation[bidder_id]["value"]
                    for bidder_id in self.efficient_allocation.keys()
                ]
            )

        mip = autoclass(self.mip_path)(self._bidder_list)
        mip.setDisplayOutput(display_output)

        allocation = mip.calculateAllocation()

        self.efficient_allocation = {}

        for bidder_id, bidder in self.population.items():
            self.efficient_allocation[bidder_id] = {}
            self.efficient_allocation[bidder_id]["good_ids"] = []
            self.efficient_allocation[bidder_id]["good_count"] = []
            bidder_allocation = allocation.allocationOf(bidder)
            good_iterator = (
                bidder_allocation.getBundle().getBundleEntries().iterator()
            )
            while good_iterator.hasNext():
                entry = good_iterator.next()
                self.efficient_allocation[bidder_id]["good_ids"].append(
                    self._good_to_id[entry.getGood().getName()]
                )
                self.efficient_allocation[bidder_id]["good_count"].append(entry.getAmount())

            self.efficient_allocation[bidder_id][
                "value"
            ] = bidder_allocation.getValue().doubleValue()

        return (
            self.efficient_allocation,
            allocation.getTotalAllocationValue().doubleValue(),
        )

    def get_uniform_random_bids(self, bidder_id, number_of_bids, seed=None):
        bidder = self.population[bidder_id]
        goods = LinkedList()
        for good in self.goods.values():
            goods.add(good)
        if seed:
            random = Random(seed)
        else:
            random = Random()

        bids = []
        for i in range(number_of_bids):
            bid = []
            bundle = bidder.getAllocationLimit().getUniformRandomBundle(random, goods)
            for good_id, good in self.goods.items():
                bid.append(bundle.countGood(self.goods[i]))
            bid.append(bidder.getValue(bundle).doubleValue())
            bids.append(bid)
        return bids

    def get_random_bids(
            self,
            bidder_id,
            number_of_bids,
            seed=None,
            mean_bundle_size=9,
            standard_deviation_bundle_size=4.5,
    ):
        bidder = self.population[bidder_id]
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()
        valueFunction = cast(
            "org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR",
            bidder.getValueFunction(SizeBasedUniqueRandomXOR, rng),
        )
        valueFunction.setDistribution(mean_bundle_size, standard_deviation_bundle_size)
        valueFunction.setIterations(number_of_bids)
        xorBidIterator = valueFunction.iterator()
        bids = []
        while xorBidIterator.hasNext():
            bundleValue = xorBidIterator.next()
            bid = []
            for good_id, good in self.goods.items():
                bid.append(bundleValue.getBundle().countGood(self.goods[good_id]))
            bid.append(bundleValue.getAmount().doubleValue())
            bids.append(bid)
        return bids

    def _vector_to_bundle(self, vector):
        assert len(vector) == len(self.goods.keys())
        bundleEntries = HashSet()
        for i in range(len(vector)):
            if vector[i] > 0:
                bundleEntries.add(BundleEntry(self.goods[i], vector[i]))
        return Bundle(bundleEntries)

    def get_model_name(self):
        return super.get_model_name() + "_generic"
    
    def get_capacities(self):
        """
        Returns a hasmap from good id -> total capacity
        """
        capacities = {i: len(self.good_to_licence[i]) for i in range(len(self.good_to_licence))}
        return capacities