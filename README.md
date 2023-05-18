# Sentence Embedder Guided Utterance Encoder (SEGUE) for Spoken Language Understanding

Main files:
 - `plots/` - scatterplot scripts for few-shot tasks
 - `segue/` - SEGUE-related classes
 - `tasks/` - downstream tasks for evaluation
   - `**/*avg.py` - task-specific checkpoint averaging script
   - `**/finetune.py` - task-specific fine-tuning script
   - `**/finetune_w2v2.py` - (if exists) same as above but for when Wav2Vec 2.0 requires different training settings than SEGUE's
   - `**/train_tl.py` - task-specific script for full-data transfer w/ a frozen backbone
   - `**/few_shot*.py` - task-specific few-shot script
 - `pretrain_avg.py` - checkpoint averaging script for the pre-trained model
 - `pretrain_segue.py` - training script for pre-training SEGUE
