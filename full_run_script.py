from TokenClassificationTrainer import TokenClassificationTrainer
import pandas as pd
import numpy as np
import torch
import pickle


def runs(discriminate_lr = False, scheduler = False, save_name = "baseline"):
    # Set the task and name of the pretrained model and the batch size for finetuning
    task = "ner"
    model_name = "xlm-mlm-17-1280"  # "bert-base-multilingual-cased" or "xlm-mlm-17-1280"
    seed = np.random.randint(0,100000)
    save_name = save_name + ".seed-" + str(seed)
    batch_size = 32

    # Flag to indicate whether to label all tokens or just the first token of each word
    label_all_tokens = True

    # File paths to splits of the chosen dataset
    file_paths = {
        "train": "data/datasets/baseline/en_ewt_nn_train_newsgroup_and_weblogs.conll",
        "validation": "data/datasets/baseline/en_ewt_nn_dev_newsgroup_and_weblogs.conll",
    }

    # initialize trainer
    trainer = TokenClassificationTrainer(task, model_name, save_name, batch_size, label_all_tokens, file_paths)


    # Training
    trainer.train(discriminate_lr = discriminate_lr, scheduler = scheduler, seed = seed)

    evals = trainer.evaluate_multiple(["data/datasets/baseline/en_ewt_nn_newsgroup_test.conll", "data/datasets/NoSta-D/NER-de-test.tsv", "data/datasets/DaNplus/da_news_test.tsv", "data/datasets/hungarian/hungarian_test.tsv"])

    baseline_eval_baseline_model = evals[0]
    NoStaD_eval_baseline_model = evals[1]
    DaNplus_eval_baseline_model = evals[2]
    Hungarian_eval_baseline_model = evals[3]

    cols = ["Dataset", "Language", "Seed"] + [name for name, _ in baseline_eval_baseline_model.items()]

    df = pd.DataFrame(columns=cols)


    trainer.del_trainer()
    del trainer
    
    # Add the evals to df
    df.loc[0] = ["Baseline", "English", seed] + [value for _, value in baseline_eval_baseline_model.items()]
    df.loc[1] = ["NoSta-D", "German", seed] + [value for _, value in NoStaD_eval_baseline_model.items()]
    df.loc[2] = ["DaNplus", "Danish", seed] + [value for _, value in DaNplus_eval_baseline_model.items()]
    df.loc[3] = ["Hungarian", "Hungarian", seed] + [value for _, value in Hungarian_eval_baseline_model.items()]
    
    
    
    
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    return df



if __name__ == "__main__":
    list_baseline = []
    list_discriminate_lr = []
    list_scheduler = []
    list_both = []
    with open('its working', 'wb') as f:
        pickle.dump('hahaha',f)
    for i in range(10):
        list_baseline.append(runs(save_name = "baseline"))
        with open('list_baseline', 'wb') as f:
            pickle.dump(list_baseline,f)
            
        list_discriminate_lr.append(runs(discriminate_lr = True, save_name = "discriminate_lr"))
        with open('list_discriminate_lr', 'wb') as f:
            pickle.dump(list_discriminate_lr,f)  
            
        list_scheduler.append(runs(scheduler = True, save_name = "scheduler"))
        with open('list_scheduler', 'wb') as f:
            pickle.dump(list_scheduler,f)       
            
        list_both.append(runs(discriminate_lr = True, scheduler = True, save_name = "both"))
        with open('list_both', 'wb') as f:
            pickle.dump(list_both,f)   
        