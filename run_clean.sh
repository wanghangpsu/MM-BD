echo ----------------------------start training ------------------------------

for((integer = 18; integer <= 9; integer ++))
do

  foo2="python train.py --model_dir clean$integer"
  $foo2
done

