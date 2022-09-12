# Coordinate regression for event-based data

The repository demonstrates coordinate regression for event-based data with spiking neural networks.
Specifically, we contribute:

1. A dataset of event-based vision (EBV) videos for coordinate regression and pose estimation
2. A method for differentiable coordinate transform (DVS) for spiking neural networks
3. Translation-invariant receptive fields that outperforms similar artificial neural network models

## Usage

To train the models, follow the below steps

1. [Download the dataset via this link](https://kth-my.sharepoint.com/:u:/g/personal/jeped_ug_kth_se/EZS0BB9N5AlAo9uB9aq0ssYB1bnFNO7JDfv1LpQTqGAy7w?e=DoFiJZ) and unpack it to a folder you can recall, say `/tmp/eventdata`.
2. Ensure you have a Python installation with [PyTorch](https://pytorch.org) and [Norse](https://github.com/norse/norse) installed.
   * After installing the necessary PyTorch version, you can install the dependencies from the `requirements.txt`-file by typing: `pip install -r requirements.txt`
3. Enter the `coordinate-regression` folder and run the `learn_shapes.py` file with the dataset directory and model type to start training
   * As an example, run `python learn_shapes.py --data_root=/tmp/eventdata --model=snn`
     * Four models are available: `ann`, `annsf`, `snn`, and `snnrf`
     * For training parameter descriptions and help, type `python learn_shapes.py --help`

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
