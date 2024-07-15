# Bail
python main.py --dataset='bail' --c_lr=0.01 --e_lr=0.01 --epochs=1500 --alpha=2 --runs=10

# German
python main.py --dataset='german' --c_lr=0.01 --e_lr=0.01 --epochs=1500 --alpha=4 --runs=10 --eta=0.7

# Credit
python main.py --dataset='credit' --c_lr=0.01 --e_lr=0.01 --c_wd=0.0001 --e_wd=0.0001 --epochs=2000 --alpha=1 --runs=10