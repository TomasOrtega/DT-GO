# DT-GO
This repository provides code for DT-GO (Delay Tolerant Gossiped Optimization).

We provide code for the main experiments in the papers: 

[**Decentralized Optimization in Networks with
Arbitrary Delays**](https://arxiv.org/abs/2401.11344)

[**Decentralized Optimization in Time-Varying Networks with Arbitrary Delays**](https://arxiv.org/abs/2405.19513)

# Running the code
Please see `requirements.txt` to install the appropriate python requirements. Scienceplots is only required to make publication-quality plots, but is not needed to run the experiments.

To reproduce the experiments in [Decentralized Optimization in Networks with
Arbitrary Delays](https://arxiv.org/abs/2401.11344), navigate to the `experiments` folder and run `python p_experiment.py` and `python lambda_experiment.py` respectively.

* The code in `p_experiment.py` will run in less than a minute in any modern laptop. 
* The code in `lambda_experiment.py` may take about 30 minutes to run in any modern laptop.

The code will produce the results for the paper figures for both experiments as-is.
To produce the figures, navigate to the `experiments` folder and run `python plot_p_experiment.py` and `python plot_lambda_experiment.py` respectively.

To reproduce the experiments in [Decentralized Optimization in Time-Varying Networks with Arbitrary Delays](https://arxiv.org/abs/2405.19513), navigate to the `experiments` folder and run `python experiment_regression.py` with the desired configurations.

To produce the figures, navigate to the `experiments` folder and run `python plot_experiment_p_err.py`, for example.

# References

If you use the code, please consider citing the following papers:

```
@misc{ortega2024decentralized_b,
      title={Decentralized Optimization in Time-Varying Networks with Arbitrary Delays}, 
      author={Tomas Ortega and Hamid Jafarkhani},
      year={2024},
      eprint={2405.19513},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@misc{ortega2024decentralized,
      title={Decentralized Optimization in Networks with Arbitrary Delays}, 
      author={Tomas Ortega and Hamid Jafarkhani},
      year={2024},
      eprint={2401.11344},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```

