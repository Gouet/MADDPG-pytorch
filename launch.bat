call C:\Users\Victor\Anaconda3\Scripts\activate.bat
call conda activate GYM_ENV_RL

python train.py --scenario simple_adversary --saved-episode 500
pause