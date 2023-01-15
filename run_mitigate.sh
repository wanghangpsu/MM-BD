


echo ----------------------------mitigate ------------------------------

for((integer = 0; integer <= 9; integer ++))
do
  foo1="python univ_bm.py --model_dir model$integer --attack_dir attack$integer"
  $foo1

done


