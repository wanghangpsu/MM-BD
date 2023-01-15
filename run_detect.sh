echo ----------------------------start detecting ------------------------------

for((integer = 0; integer <= 9; integer ++))
do
  foo1="python univ_bd.py --model_dir model$integer"
  $foo1
done

