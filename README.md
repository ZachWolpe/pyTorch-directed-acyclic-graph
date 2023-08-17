# PyTorch’s Dynamic Graphs (Autograd)

![Gradient propagation Data Structures](https://github.com/ZachWolpe/pyTorch-directed-acyclic-graph/blob/main/neural_net_schematic/expr-run.webp)

The acyclical graphs design powering modern deep learning frameworks. `PyTorch` (and `Tensorflow`) build dynamic graphs at runtime, allowing for the creation of complex, dynamic models. Notably, this saves RAM space, reduces setup complexity (by needing to declare the entire graph before training); allows for control flow/conditionals & allows for the creation of dynamic models (e.g. RNNs, GANs, etc.).

This is the supporting code, the full article is available here)[https://medium.com/@zachcolinwolpe/pytorchs-dynamic-graphs-autograd-96ecb3efc158]


## Summary

PyTorch’s dynamic computational graph offers a number of advantages:

- _*Flexibility*_: Allows for the creation of complex, dynamic models that contain control flow, domain/business logic & variable length parameters.
- _*Memory Efficiency*_: Constructing the graphs iteratively allows PyTorch to free up memory from previously used but now superfluous graph components — only storing what is essential in a given run.
- _*Debugging& Logging*_: Access to real-time data is valuable when debugging or monitoring a program's behaviours.
- _*Gradient toggling*_: Gradient tracking and computation can be toggled on and off at runtime. The torch.no_grad() context manager can be used during inference or to freeze a (sub)set of parameters for transfer learning, meta-learning etc.
- _*Dynamic models*_: The flexibility also allows the same framework to extend to models with variable parameter, input & output spaces.
- _*Cleaner code*_: A static graph would require more Boilerplate Code to define the gradient traversal.
- __*Ease of Experimentation*__: Prototyping is straightforward.