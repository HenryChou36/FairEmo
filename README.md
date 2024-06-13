# An Inter-Speaker Fairness-Aware Speech Emotion Regression Framework
This is the offical implemntation of An Inter-Speaker Fairness-Aware Speech Emotion Regression Framework for training speech emotion recognizer with inter-speaker fairness constraint.
The underline speech emotion recognition model is based on [MetricAug](https://github.com/crowpeter/MetricAug).
## 1. Environment Setup
```
create env -n fair_emo python==3.8
conda activate fair_emo
pip install scikit-learn  
pip install joblib  
pip install pandas  
pip install tqdm  
pip lnstall librosa  
pip install soundfile  
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install hdbscan
# for feature extract only
pip install fairseq
pip install git+https://github.com/speechbrain/speechbrain.git@develop
```
## 2. Prepare Dataset
Currently, we have implemented on following dataset.

**MSP-Pocast**: <https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html>  
**IEMOCAP**: <https://sail.usc.edu/iemocap/>
## 3. Data Preprocessing
Following `feature_extract/msp_extract.ipynb` for **MSP-Pocast** dataset or `feature_extract/iemocap_extract.ipynb` for **IEMOCAP** dataset.
## 4. Training
Modify train.py and run following command to train the model. The exp_name specify where you want the checkpoints and result information being stored.
```
python baseline.py --exp_name='exp/IEMOCAP/hubert/baseline_32_nu/CCC/All' --target_name='All'
```
The process would also inference on the testing dataset. You can inspect `info_ckpt.pkl` to do further analysis with raw prediction.
