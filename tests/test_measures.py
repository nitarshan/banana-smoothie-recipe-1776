import unittest

from ccm.dataset_helpers import DatasetType
from ccm.experiment_config import ComplexityType
from ccm.measures import get_all_measures, get_single_measure
from ccm.models import NiN

# Test that get_single_measure and get_all_measures return the same values
class MeasuresTest(unittest.TestCase):
  def test_single_all_equivalence(self):
    # Run experiment
    model = NiN(3, 4, 25, DatasetType.CIFAR10)
    init_model = NiN(3, 4, 25, DatasetType.CIFAR10)

    measures = {}
    for measure in ComplexityType:
      if not (measure == ComplexityType.NONE or measure in ComplexityType.data_dependent_measures() or measure in ComplexityType.acc_dependent_measures()):
        measures[measure] = get_single_measure(model, init_model, measure)
    all_measures = get_all_measures(model, init_model, use_reparam=False)

    for key in measures.keys():
      assert type(all_measures[key]) in {int, float}
      assert measures[key].item() == all_measures[key], f'{key} {measures[key].item()} {all_measures[key]}'

if __name__=='__main__':
  unittest.main()
