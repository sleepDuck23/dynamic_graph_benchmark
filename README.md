# Final project for Deep Learning - Master MVA
This final projects consist in learn about the models of deep learning for dynamical graphs. Using the repository constructed for the paper bellow, it was possible to understand more about the models and how they perform the learning part with respect to graphs that are constantly changing. The objective of this project was to implement the benchmark by adding models that were described on the paper and a new dataset. With this implementations the goal behing it is to have a more broad benchmark and a better understanting of those models. The project was done by Bruno A. de Araujo.

### Contributions
This is a final project for a course in a master, so there is not much time to give a great contribution, for that reason the implementations kept itself with the spatio-temporal and discrete graphs (D-TDG sector of the repository). Also there is not a major change on the repository, so the initial guidelines to run it were kept as the autor of the original paper/repository recommended.

- New dataset for the D-TDG spatio-temporal graphs: https://citibikenyc.com/system-data

This dataset changes from the idea of the others implemented on the paper, where everyone is related to traffic. The idea was t change it and see if it could lead to different behaviors from the models evaluation.

- New models for the D-TDG discrete graphs:
stacked architecture:
https://github.com/snap-stanford/roland
https://github.com/geopanag/pandemic_tgnn

random walk:
https://github.com/dev-jwel/TiaRa

Autoencoder:
https://github.com/palash1992/DynamicGEM

The idea was to implement more models that were discussed on the paper but were not implement on the benchmark, the goal is to get a better view of comparison between architectures

## Deep learning for dynamic graphs: models and benchmarks

Official code repository for our paper [***"Deep learning for dynamic graphs: models and benchmarks"***](https://ieeexplore.ieee.org/document/10490120) accepted at the IEEE Transactions on Neural Networks and Learning Systems.

Please consider citing us

	@article{gravina2024benchmark,
	    author={Gravina, Alessio and Bacciu, Davide},
        journal={IEEE Transactions on Neural Networks and Learning Systems}, 
        title={{Deep Learning for Dynamic Graphs: Models and Benchmarks}}, 
        year={2024},
        volume={},
        number={},
        pages={1-14},
        keywords={Surveys;Representation learning;Benchmark testing;Laplace equations;Graph neural networks;Message passing;Convolution;Benchmark;deep graph networks (DGNs);dynamic graphs;graph neural networks (GNNs);survey;temporal graphs},
        doi={10.1109/TNNLS.2024.3379735}
	}



### How to run the experiments
To reproduce the experiments please refer to:

- [D-TDG/README.md](https://github.com/gravins/dynamic_graph_benchmark/tree/main/D-TDG) to reproduce the experiments on the *Discrete-Time Dynamic Graph* domain. 
- [C-TDG/README.md](https://github.com/gravins/dynamic_graph_benchmark/tree/main/C-TDG) to reproduce the experiments on the *Continuous-Time Dynamic Graph* domain. 


