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
        "Discriminate": "list_discriminate_lr",
    }

    seeds = dict()

    for name, path in paths.items():
        with open(f"{path}", "rb") as f:
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
        "Discriminate": "list_discriminate_lr",
    }

    results = {
        "english": {
            "Baseline": [],
            "Discriminate": [],
        },
        "german": {
            "Baseline": [],
            "Discriminate": [],
        },
        "danish": {
            "Baseline": [],
            "Discriminate": [],
        },
        "hungarian": {
            "Baseline": [],
            "Discriminate": [],
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

                if metric == "eval_loss":
                    result = -result

                if result == 0:
                    print(run)

                results[lang][name].append(result)
        
        print(f"Read {name} results")
    
    return results


def evaluate_aso(path="eval_lists", metric="span_f1", seed=123, **kwargs):

    results = read_results(path, metric=metric)
    my_model_scores_per_dataset = []
    baseline_scores_per_dataset = []
    for lang in results.keys():
        print(f"Language: {lang}")
        
        for name, result in results[lang].items():
            print(f"{name}: {np.mean(result):.4f} +- {np.std(result): .4f}")
        print((np.mean(results[lang]['Baseline']) -  np.mean(results[lang]['Discriminate'])) / np.mean(results[lang]['Baseline']) * 100)
        my_model_scores_per_dataset.append(results[lang]['Discriminate'])
        baseline_scores_per_dataset.append(results[lang]['Baseline'])

        # aso_result = aso(results[lang]['Discriminate'], results[lang]['Baseline'], confidence_level=0.9875, seed=seed, **kwargs)

        # print(f"ASO: \n{aso_result}")

    eps_min = [aso(b, a, confidence_level=0.95, num_comparisons=4, seed=seed) for a, b in zip(my_model_scores_per_dataset, baseline_scores_per_dataset)]
    print(f"eps_min: {eps_min}")
    return


def main():
    # pprint(read_results("pickled_evals"))
    # pprint(check_seeds())
    evaluate_aso(metric="span_f1", seed=123)

    results = read_results("eval_lists", metric="span_f1")

    fig = plt.figure(figsize=(10, 10))
    for i, lang in enumerate(results.keys()):
        plt.subplot(2, 2, i+1)
        plt.title(lang)
        for name, result in results[lang].items():
            mu = np.mean(result)
            sigma = np.std(result)

            # plot kde of results
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            kde = stats.gaussian_kde(result)
            plt.plot(x, kde(x), label=name)

    
        
        plt.legend()
    
    plt.show()






if __name__ == '__main__':
    main()