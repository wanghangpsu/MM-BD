# MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic

This is the implementation of the IEEE S&P 2024 paper: MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic

The mitigation method was the second place at the first IEEE Trojan Removal Competition (https://www.trojan-removal.com/?page_id=2)

This repository includes:
- Training code for the clean model and attacked model.
- MM-BD backdoor detection code.
- MM-BM backdoor mitigation code.



## Requirements
Ubuntu 20.04
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

For attack models (BadNet in the paper)
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
## Questions&issues

if you run into any issues in running the experiments (e.g. on different model architectures), or have any questions, please contact wanghangpsu@gmail.com for help!

## <a name="Citation"></a>Citation

Please consider citing our work if it helps your research.
```bib
@inproceedings{MM-BD,
    title={MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic},
    author={Wang, Hang and Xiang, Zhen and Miller, David J and Kesidis, George},
    booktitle={IEEE Symposium on Security and Privacy},
    year={2024},
}
```
