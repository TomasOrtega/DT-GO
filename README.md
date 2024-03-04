# DT-GO
This repository provides code for DT-GO (Delay Tolerant Gossiped Optimization).

We provide code for the main experiments in the paper: **Decentralized Optimization in Networks with
Arbitrary Delays**

# Running the code
Please see `requirements.txt` to install the appropriate python requirements. Scienceplots is only required to make publication-quality plots, but is not needed to run the experiments.

To reproduce the experiments, navigate to the `experiments` folder and run `python p_experiment.py` and `python lambda_experiment.py` respectively.

* The code in `p_experiment.py` will run in less than a minute in any modern laptop. 
* The code in `lambda_experiment.py` may take about 30 minutes to run in any modern laptop.

The code will produce the results for the paper figures for both experiments as-is.
To produce the figures, navigate to the `experiments` folder and run `python p_experiment.py` and `python lambda_experiment.py` respectively.

# References

If you use the code, please cite the following paper:

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

