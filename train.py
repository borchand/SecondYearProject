import argparse
import numpy as np
from TokenClassificationTrainer import TokenClassificationTrainer

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--discriminative_lr", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=int, default=2e-5)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=np.random.randint(0, 1000))
args = parser.parse_args()


# Set the task and name of the pretrained model and the batch size for finetuning
task = "ner"
model_name = "xlm-mlm-17-1280"

save_name = ""

if args.discriminative_lr:
    save_name = "discriminative-lr"

if save_name == "":
    save_name = "baseline"

print(save_name)

batch_size = args.batch_size

# Flag to indicate whether to label all tokens or just the first token of each word
label_all_tokens = True

# File paths to splits of the chosen dataset
file_paths = {
    "train": "data/datasets/NoSta-D/NER-de-train.tsv",
    "validation": "data/datasets/NoSta-D/NER-de-dev.tsv",
}

# initialize trainer
trainer = TokenClassificationTrainer(task, model_name, save_name, batch_size, label_all_tokens, file_paths)

# Training
trainer.train_and_save(
    discriminate_lr = args.discriminative_lr, 
    num_epochs=args.num_epochs, 
    learning_rate=args.lr,
    seed=args.seed
    )

