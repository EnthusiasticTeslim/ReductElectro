## ***************** 1st run ***************** ##
# id = 0
python src/create_model.py --data_path '../data/data.xlsx' --label 'C_2H_4' --label_index 0 --reshuffle --seed 200009 --save
# id = 1
python src/create_model.py --data_path '../data/data.xlsx' --label 'CO' --label_index 1 --seed 16 --save
# id = 2
python src/create_model.py --data_path '../data/data.xlsx' --label 'H_2' --label_index 2 --seed 50 --save
# id = 3, in progress
# python src/create_model.py --data_path '../data/data.xlsx' --label 'Ethanol' --label_index 3 --seed 100 --save
# id = 4
python src/create_model.py --data_path './data/data.xlsx' --label 'Formate' --label_index 4 --seed 22 --save

## ***************** 2nd run ***************** ##
# id = 0
python src/create_model.py --data_path '../data/data_v2.xlsx' --label 'C_2H_4' --label_index 0 --reshuffle --seed 200009
# id = 1
python src/create_model.py --data_path '../data/data_v2.xlsx' --label 'CO' --label_index 1 --seed 16 --save
# id = 2, in progress
python src/create_model.py --data_path '../data/data_v2.xlsx' --label 'H_2' --label_index 2 --seed 25 --reshuffle
# id = 3, in progress
# python src/create_model.py --data_path '../data/data_v2.xlsx' --label 'Ethanol' --label_index 3 --seed 100 --save
# id = 4
python src/create_model.py --data_path '../data/data_v2.xlsx' --label 'Formate' --label_index 4 --seed 22 --save