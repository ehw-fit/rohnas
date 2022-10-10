# RoHNAS - neural architecture search for adversial-attack resilient Capsule neural networks

A Neural Architecture Search Framework with Conjoint Optimization for Adversarial Robustness and Hardware Efficiency of Convolutional and Capsule Networks. This repository provides source codes from neural architecture search of convolutional capsule networks with respect to the adversial attack robustness. For more detail please follow [our paper](tbd). If you used these results in your research, please refer to the paper.


     MARCHISIO Alberto, MRAZEK Vojtech, MASSA Andrea, BUSSOLINO Beatrice, MARTINA Mauricio a SHAFIQUE Muhammad. RoHNAS: A Neural Architecture Search Framework with Conjoint Optimization for Adversarial Robustness and Hardware Efficiency of Convolutional and Capsule Networks. IEEE Access, 2022. ISSN 2169-3536.



```bibtex
@INPROCEEDINGS{rohnas:2022,
   author = "Alberto Marchisio and Vojtech Mrazek and Andrea Massa and Beatrice Bussolino and Mauricio Martina and Muhammad Shafique",
   title = "RoHNAS: A Neural Architecture Search Framework with Conjoint Optimization for Adversarial Robustness and Hardware Efficiency of Convolutional and Capsule Networks",
   journal = "IEEE Access",
   year = 2022,
   ISSN = "2169-3536",
}
```

If you found any problem or something in the description is not clear, please feel free to create an issue ticket.

## Installation of the testing environment

The testing environment requires TensorFlow. Moreover, a Pareto-frontier package must be installed.

    pip3 install --user git+https://github.com/ehw-fit/py-paretoarchive


### Instalation using Anaconda
```bash
conda env create -n tf-1.13-gpu -f environment.yml


source activate tf-1.13-gpu
pip install --user git+https://github.com/ehw-fit/py-paretoarchive

# do the commands
conda deactivate
```



## Using a scripts

All executable scripts are located in "nsga" folder. You can use `-h` parameter to get a list of available parameters. Please note that not all parameters must be used in the scripts. See examples bellow

 * `main.py` runs NSGA NAS search 
 * `randlearn.py` generates random neural networks and evaluates them
 * `chreval.py` evaluates the results from NAS for a larger amount of epochs
 * `chreval_complex.py` a different training algorithm optimized for CIFAR-10 for evaluation

### Training 

```bash 
work="results" # working directory for saving results

source activate tf-1.13-gpu # activating valid environment with Anaconda

cd nsga

dat="mnist-$(date +"%F-%H-%M")"
python main.py --epochs 5 \   # number of epochs
	--output out_${dat} \     # 
	--population 10 --offsprings 10 \  #nsga settings
	--generations 50 \   # 
        --timeout 300 \  # timeout in in seconds for training of one candidate
	--eps 0.003 0.1 \    # selected epsilon levels
	--save_dir ${work}/data \  # directory with results
	--cache_dir ${work}/cache \ # 
	2>${work}/logs/${dat}.err > ${work}/logs/${dat}.std
```


```sh
python randlearn.py --epochs 10 \
	--dataset cifar10 \
	--output "${work}/rand_short/cifar/${dat}" \
	--eps 0.003 0.1 \    # selected epsilon levels
	--max_params 0 \
        --timeout 600 \
	--save_dir ${work}/data \
	--cache_dir ${work}/cache \
	2>${work}/rand_short/cifar/${dat}.err > ${work}/rand_short/cifar/${dat}.std &
```


### Testing
```sh
cd nsga
python chrlearn.py --epochs 150 \
	--output "./logrand/${dat}" \
	--dataset mnist \
	--max_params 0 \
	--eps 0.001 0.03 0.01 \
	"$fn"                       \
	2>../logrand/${dat}.err > ../logrand/${dat}.std
```

```sh
ds="cifar100"
PARAMS="--batch_size 100 --epochs 300 --shift_fraction 0.078125 --rotation_range 2 --horizontal_flip --lr 0.001 --lr_decay 0.96 --decay_steps 6000 --lam_recon 0.005 "
python chrlearn_complex.py --dataset ${ds} --epochs 300 $PARAMS ../data/chrom_deepcaps.chr 1>${outdir}/${ds}-deepcaps.std.log 2>${outdir}/${ds}-deepcaps.err.log 
```
