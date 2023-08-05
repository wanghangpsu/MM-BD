# MMBD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic

This is the implementation of the paper: MMBD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic

The mitigation method was the second place at the first IEEE Trojan Removal Competition (https://www.trojan-removal.com/?page_id=2)

This repository includes:
- Training code for the clean model and attacked model.
- MMBD backdoor detection code.
- MMMB backdoor mitigation code.



## Requirements
Ubuntu 20.24
Python 3.7
- Install required python packages:
```bash
$ pip install numpy
$ pip install torch
$ pip install torchvision
$ pip install matplotlib
$ pip install scipy
$ pip install pillow
```


## Training
For clean model training,
run command:
```bash
$ ./run_clean.sh
```
Which gives 10 clean models saved in ./clean0 to ./clean9 folders

For attack models (A3-M in the paper)
run_command:
```bash
$ ./run_attack.sh
```

Which gives 10 attacked modes saved in ./model0 to ./model9 folders

## MMBD Detection
Run_command:
```bash
$ ./run_detect.sh
```
Which applies the UnivBD method on all the models. 


## MMMB Mitigation

Run:
```bash
$ ./run_mitigate.sh
```
