import argparse
import pandas as pd
from TokenClassificationTrainer import TokenClassificationTrainer
import os

def eval(task, model_name, save_name, batch_size):
    # Flag to indicate whether to label all tokens or just the first token of each word
    label_all_tokens = True

    # File paths to splits of the chosen dataset
    file_paths = {
        "train": "data/datasets/baseline/en_ewt_nn_train.conll",
        "validation": "data/datasets/baseline/en_ewt_nn_train.conll",
    }

    trainer = TokenClassificationTrainer(task, model_name, save_name, batch_size, label_all_tokens, file_paths)

    # load trianed model to trainer
    trainer.set_trainer(use_old = True)
    baseline_eval, NoStaD_eval, DaNplus_eval, Hungarian_eval  = trainer.evaluate_multiple(["data/datasets/baseline/en_ewt_nn_newsgroup_test.conll", "data/datasets/NoSta-D/NER-de-test.tsv", "data/datasets/DaNplus/da_news_test.tsv", "data/datasets/hungarian/hungarian_test.tsv"])

    cols = ["Dataset", "Language"] + [name for name, _ in baseline_eval.items()]

    df = pd.DataFrame(columns=cols)

    # Add the evals to df
    df.loc[0] = ["Baseline", "English"] + [value for _, value in baseline_eval.items()]
    df.loc[1] = ["NoSta-D", "German"] + [value for _, value in NoStaD_eval.items()]
    df.loc[2] = ["DaNplus", "Danish"] + [value for _, value in DaNplus_eval.items()]
    df.loc[3] = ["Hungarian", "Hungarian"] + [value for _, value in Hungarian_eval.items()]

    return df

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--discriminative_lr", type=bool, default=False)
    parser.add_argument("--cosine_schedule", type=bool, default=False)
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--to_csv", type=bool, default=True)
    args = parser.parse_args()

    # Set the task and name of the pretrained model and the batch size for finetuning
    task = "ner"
    model_name = "xlm-mlm-17-1280"

    save_name = args.save_name

    if save_name == "":
        if args.cosine_schedule:
            save_name = "scheduler"

        if args.discriminative_lr:
            if save_name == "":
                save_name = "discriminative-lr"
            else:
                save_name += "_AND_discriminative_lr"

        if save_name == "":
            save_name = "baseline"

    df = eval(task, model_name, save_name, args.batch_size)
    if args.to_csv:
        if not os.path.exists('./evaluations'):
            os.makedirs('./evaluations')
        df.to_csv(f"evaluations/{save_name}.csv", index=False)
    else:
        print(df)
