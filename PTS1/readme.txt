- please make sure ${PWD} is in the unzipped folder
- data fold: saving PTS1 dataset for training and predicting
- main fold: the two models are applied step-by-step. 
	- first run agent_1.py, whose input files and their paths are listed in agent1.config.yaml

	- then run agent_2.py, whose input files and their paths are listed in agent2.config.yaml


To run the code, simply go into the main fold.
	| python ../main/agent_1.py
	| python ../main/agent_2.py
	| python ../main/venn4_no_old.py
	| python ../main/venn4_old.py

To see the prediction result, see /data/pts1/result/
	- rest_prediction_pos.csv is the prediction from agent1 (LDA), including positive and negative predictions for the rest of 8k data points
	- classification_top1000.csv is the prediction from agent2 (SVC), including positive and negative predictions for the rest of 8k data points

