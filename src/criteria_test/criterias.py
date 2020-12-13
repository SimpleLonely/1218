import sys
sys.path.append("C:\\WorkSpace\\1218\\src\\criteria_test")
from coverage_criteria import neuron_coverage,multi_testing_criteria

def get_criterias():
    criterias = {}
    criterias['neuron_coverage'] = neuron_coverage.neuron_coverage("gold","GoldModel","test")
    [KMN, NB, SNA, TKNC, TKNP] = multi_testing_criteria.multi_testing_criteria("gold","GoldModel","test",0.0,1000,2)
    criterias['KMN'] = KMN
    criterias['NB'] = NB
    criterias['SNA'] = SNA
    criterias["TKNC"] = TKNC
    criterias['TKNP'] = TKNP
    return criterias