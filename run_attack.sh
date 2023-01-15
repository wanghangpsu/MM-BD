echo ----------------------------start training ------------------------------

for((integer = 0; integer <= 9; integer ++))
do
  foo1="python attacks_crafting.py --out_dir attack$integer"
  $foo1
  foo2="python train_models_contam.py --attack_dir attack$integer --model_dir model$integer"
  $foo2
done

