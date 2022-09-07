# Coordinate regression for event-based data

The repository demonstrates coordinate regression for event-based data with spiking neural networks.
Specifically, we contribute:

1. A dataset of event-based vision (EBV) videos for coordinate regression and pose estimation
2. A method for differentiable coordinate transform (DVS) for spiking neural networks
3. Translation-invariant receptive fields that outperforms similar artificial neural network models

## Usage

All work here can be reproduced as follows

1. Download the dataset by contacting the author (will be uploaded soon)
2. Ensure you have a Python installation with [PyTorch](https://pytorch.org) and [Norse](https://github.com/norse/norse) installed.
   1. After installing the necessary PyTorch version, you can install the dependencies from the `requirements.txt`-file by typing: `pip install -r requirements.txt`
3. Run the `learn_shapes.py` file to train the model

## Authors and Contact

* Jens E. Pedersen `<jeped@kth.se>` ([Twitter @jensegholm](https://twitter.com/jensegholm))
* Juan P. Romero B.
* Jörg Conradt

## Acknowledgements

This work has been performed at the
[Neurocomputing Systems Lab](https://neurocomputing.systems) at
[KTH Royal Institute of Technology](https://kth.se) and funded by the
[Human Brain Project](https://www.humanbrainproject.eu/) and the
[AI Pioneer Centre](https://www.aicentre.dk).

## License
This work is licensed under LGPLv3.
