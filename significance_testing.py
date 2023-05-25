import pickle
import numpy as np
import glob
from pprint import pprint
import matplotlib.pyplot as plt
from scipy import stats

from deepsig import aso, multi_aso


def check_seeds():

    paths = {
        "Baseline": "list_baseline",
        "Scheduler": "list_scheduler",
        "Discriminate": "list_discriminate_lr",
        "Both": "list_both"
    }

    seeds = dict()

    for name, path in paths.items():
        with open(f"pickled_evals/{path}", "rb") as f:
            runs = pickle.load(f)
        
        seeds[name] = []

        for run in runs:
            seeds[name].append(run.at[0, "Seed"])
        
        print(f"{name}: {len(set(seeds[name]))}")
    
    return seeds



def read_results(folder_path, metric="span_f1"):
    # Use glob to get all files in path
    paths = {
        "Baseline": "list_baseline",
        "Scheduler": "list_scheduler",
        "Discriminate": "list_discriminate_lr",
        "Both": "list_both"
    }

    results = {
        "english": {
            "Baseline": [],
            "Scheduler": [],
            "Discriminate": [],
            "Both": []
        },
        "german": {
            "Baseline": [],
            "Scheduler": [],
            "Discriminate": [],
            "Both": []
        },
        "danish": {
            "Baseline": [],
            "Scheduler": [],
            "Discriminate": [],
            "Both": []
        },
        "hungarian": {
            "Baseline": [],
            "Scheduler": [],
            "Discriminate": [],
            "Both": []
        }
    }

    metric = f"eval_{metric}"

    for name, path in paths.items():
        with open(f"{folder_path}/{path}", "rb") as f:
            runs = pickle.load(f)
        
        for j, run in enumerate(runs):
            if j == 7:
                continue
            for i, lang in enumerate(results.keys()):
                result = run.at[i, metric]

                if result == 0:
                    print(run)

                results[lang][name].append(result)
        
        print(f"Read {name} results")
    
    return results


def evaluate_aso(path="pickled_evals", metric="span_f1", seed=42, **kwargs):

    results = read_results(path, metric=metric)

    for lang in results.keys():
        print(f"Language: {lang}")
        
        for name, result in results[lang].items():
            print(f"{name}: {np.mean(result)}")
        
        aso_result = multi_aso(results[lang], confidence_level=0.95, return_df=True, seed=seed, **kwargs)

        print(f"ASO: \n{aso_result}")

        print()

    
    return


    # Evaluate using ASO
    aso_result = multi_aso(results, confidence_level=0.95, return_df=True, seed=seed, **kwargs)

    print(f"ASO: {aso_result}")


def main():
    # pprint(read_results("pickled_evals"))
    # pprint(check_seeds())
    # evaluate_aso(metric="span_f1", seed=42)

    results = read_results("pickled_evals", metric="span_f1")

    fig = plt.figure(figsize=(10, 10))
    for i, lang in enumerate(results.keys()):
        plt.subplot(2, 2, i+1)
        plt.title(lang)
        mus = dict()
        std = dict()
        for name, result in results[lang].items():
            mus[name] = np.mean(result)
            std[name] = np.std(result)
        
        # Plot cdf of normal distribution for each
        for name, mu in mus.items():
            x = np.linspace(mu - 3*std[name], mu + 3*std[name], 100)
            plt.plot(x, stats.norm.pdf(x, mu, std[name]), label=name)
        
        plt.legend()
    
    plt.show()






if __name__ == '__main__':
    main()