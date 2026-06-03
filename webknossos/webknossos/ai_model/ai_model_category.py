from enum import Enum, unique


@unique
class AiModelCategory(Enum):
    em_neurons = "em_neurons"
    em_nuclei = "em_nuclei"
    em_synapses = "em_synapses"
    em_neuron_types = "em_neuron_types"
    em_cell_organelles = "em_cell_organelles"
