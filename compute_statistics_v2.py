import statistics
from matplotlib import pyplot as plt 

import sys
import subprocess
import os
import numpy as np
import argparse



def first_plot(data_to_plot, dataset_name, k):

    values_so_max = [data_to_plot["corre{}-{}-so.txt".format(i, dataset_name)][2] for i in range(2,k) ]
    values_mo_max = [data_to_plot["corre{}-{}-mo.txt".format(i, dataset_name)][2] for i in range(2,k) ]

    values_so = [data_to_plot["corre{}-{}-so.txt".format(i, dataset_name)][0] for i in range(2,k) ]
    values_mo = [data_to_plot["corre{}-{}-mo.txt".format(i, dataset_name)][0] for i in range(2,k) ]

    values_so_sdv = [np.sqrt(data_to_plot["corre{}-{}-so.txt".format(i, dataset_name)][1]) for i in range(2,k) ]
    values_mo_sdv = [np.sqrt(data_to_plot["corre{}-{}-mo.txt".format(i, dataset_name)][1]) for i in range(2,k) ]

    plt.xticks(range(2,k))

    plt.axhline(y = data_to_plot["original-{}.txt".format(dataset_name)][0], color = 'm', linestyle = '-', label="Original (avg)")     # non isomorphic graphs set at the beginning of the notebook
    plt.axhline(y = data_to_plot["original-{}.txt".format(dataset_name)][2], color = 'm', linestyle = '--', label="Original (max)")     # non isomorphic graphs set at the beginning of the notebook


    plt.plot( [i for i in range(2,k)], values_so_max, label='SO (max)', linestyle = '--', color = 'g')
    plt.plot( [i for i in range(2,k)], values_mo_max, label='MO (max)', linestyle = '--', color = 'b')

    plt.errorbar([i for i in range(2,k)], values_so, yerr=values_so_sdv, ecolor='lightgray', elinewidth=3, capsize=1, label='SO (avg)', linestyle = '-', color = 'g')
    plt.errorbar([i for i in range(2,k)], values_mo, yerr=values_mo_sdv, ecolor='lightblue', elinewidth=3, capsize=1, label='MO (avg)', linestyle = '-', color = 'b')


    plt.title("Accuracies of different models on {} dataset".format(dataset_name))

    plt.legend(ncol=2, loc='lower left')
    plt.savefig('accuracies-{}.png'.format(dataset_name))
    plt.figure()

def second_plot(data_to_plot, dataset_name, k):

    values_so_max_PR = [data_to_plot["corre{}-{}-PR-so.txt".format(i, dataset_name)][2] for i in range(2,k) ]
    values_mo_max_PR = [data_to_plot["corre{}-{}-PR-mo.txt".format(i, dataset_name)][2] for i in range(2,k) ]
    values_so_PR = [data_to_plot["corre{}-{}-PR-so.txt".format(i, dataset_name)][0] for i in range(2,k) ]
    values_mo_PR = [data_to_plot["corre{}-{}-PR-mo.txt".format(i, dataset_name)][0] for i in range(2,k) ]
    values_so_PR_sdv = [np.sqrt(data_to_plot["corre{}-{}-PR-so.txt".format(i, dataset_name)][1]) for i in range(2,k) ]
    values_mo_PR_sdv = [np.sqrt(data_to_plot["corre{}-{}-PR-mo.txt".format(i, dataset_name)][1]) for i in range(2,k) ]

    plt.xticks(range(2,k))

    plt.axhline(y = data_to_plot["original-{}.txt".format(dataset_name)][0], color = 'm', linestyle = '-', label="Original (avg)")     # non isomorphic graphs set at the beginning of the notebook
    plt.axhline(y = data_to_plot["original-{}.txt".format(dataset_name)][2], color = 'm', linestyle = '--', label="Original (max)")     # non isomorphic graphs set at the beginning of the notebook
    # plt.axhline(y = plot_me["original-{}-PR.txt".format(dataset_name)][0], color = 'black', linestyle = '-', label="Original (avg)")     # non isomorphic graphs set at the beginning of the notebook
    # plt.axhline(y = plot_me["original-{}-PR.txt".format(dataset_name)][2], color = 'black', linestyle = '--', label="Original (max)")     # non isomorphic graphs set at the beginning of the notebook



    plt.plot( [i+1 for i in range(2,k)], values_so_max_PR, label='SO (max) - PR=0.2', linestyle = '--', color = 'g')
    plt.plot( [i+1 for i in range(2,k)], values_mo_max_PR, label='MO (max) - PR=0.2', linestyle = '--', color = 'b')

    plt.errorbar([i+1 for i in range(2,k)], values_so_PR, yerr=values_so_PR_sdv, ecolor='lightgray', elinewidth=3, capsize=1, label='SO (avg) - PR=0.2', linestyle = '-', color = 'g')
    plt.errorbar([i+1 for i in range(2,k)], values_mo_PR, yerr=values_mo_PR_sdv, ecolor='lightblue', elinewidth=3, capsize=1, label='MO (avg) - PR=0.2', linestyle = '-', color = 'b')


    plt.title("Accuracies of different models on {} dataset".format(dataset_name))
        # plt.yscale('log', base=2)  # Set the scale of the y-axis to logarithmic

        # plt.title("Time needed in hrs")
    plt.legend(ncol=2, loc='lower left')
    plt.savefig('accuracies-second-{}.png'.format(dataset_name))
    plt.figure()


def third_plot(data_to_plot, dataset_name, k):
    values_so_max_PR = [data_to_plot["corre{}-{}-M1-so.txt".format(i, dataset_name)][2] for i in range(2,k) ]
    values_mo_max_PR = [data_to_plot["corre{}-{}-M1-mo.txt".format(i, dataset_name)][2] for i in range(2,k) ]
    values_so_PR = [data_to_plot["corre{}-{}-M1-so.txt".format(i, dataset_name)][0] for i in range(2,k) ]
    values_mo_PR = [data_to_plot["corre{}-{}-M1-mo.txt".format(i, dataset_name)][0] for i in range(2,k) ]
    values_so_PR_sdv = [np.sqrt(data_to_plot["corre{}-{}-M1-so.txt".format(i, dataset_name)][1]) for i in range(2,k) ]
    values_mo_PR_sdv = [np.sqrt(data_to_plot["corre{}-{}-M1-mo.txt".format(i, dataset_name)][1]) for i in range(2,k) ]

    plt.xticks(range(2,k))

    plt.axhline(y = data_to_plot["original-{}.txt".format(dataset_name)][0], color = 'm', linestyle = '-', label="Original (avg)")     # non isomorphic graphs set at the beginning of the notebook
    plt.axhline(y = data_to_plot["original-{}.txt".format(dataset_name)][2], color = 'm', linestyle = '--', label="Original (max)")     # non isomorphic graphs set at the beginning of the notebook
    # plt.axhline(y = plot_me["original-{}-PR.txt".format(dataset_name)][0], color = 'black', linestyle = '-', label="Original (avg)")     # non isomorphic graphs set at the beginning of the notebook
    # plt.axhline(y = plot_me["original-{}-PR.txt".format(dataset_name)][2], color = 'black', linestyle = '--', label="Original (max)")     # non isomorphic graphs set at the beginning of the notebook



    plt.plot( [i+1 for i in range(2,k)], values_so_max_PR, label='SO (max) - M1', linestyle = '--', color = 'g')
    plt.plot( [i+1 for i in range(2,k)], values_mo_max_PR, label='MO (max) - M1', linestyle = '--', color = 'b')

    plt.errorbar([i+1 for i in range(2,k)], values_so_PR, yerr=values_so_PR_sdv, ecolor='lightgray', elinewidth=3, capsize=1, label='SO (avg) - M1', linestyle = '-', color = 'g')
    plt.errorbar([i+1 for i in range(2,k)], values_mo_PR, yerr=values_mo_PR_sdv, ecolor='lightblue', elinewidth=3, capsize=1, label='MO (avg) - M1', linestyle = '-', color = 'b')


    plt.title("Accuracies of different models on {} dataset".format(dataset_name))
        # plt.yscale('log', base=2)  # Set the scale of the y-axis to logarithmic

        # plt.title("Time needed in hrs")
    plt.legend(ncol=2, loc='lower left')
    plt.savefig('accuracies-third-{}.png'.format(dataset_name))
    plt.figure()

def only_max_lines(data_to_plot, dataset_name, k):

    values_so_max = [data_to_plot["corre{}-{}-so.txt".format(i, dataset_name)][2] for i in range(2,k) ]
    values_mo_max = [data_to_plot["corre{}-{}-mo.txt".format(i, dataset_name)][2] for i in range(2,k) ]

    plt.figure(figsize=(10, 6))
    plt.xticks(range(2,k), fontsize=12)
    plt.yticks(fontsize=12)

    plt.axhline(y = data_to_plot["original-{}.txt".format(dataset_name)][2], color = 'm', linestyle = '--', label="Original (best)")

    plt.plot(range(2, k), values_so_max, label='SO (best)', linestyle='--', color='g', marker='o')
    plt.plot(range(2, k), values_mo_max, label='MO (best)', linestyle='--', color='b', marker='s')

    plt.xlabel('k', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title("Best Accuracies on {} Dataset".format(dataset_name), fontsize=16)

    plt.legend(ncol=2, loc='lower left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('accuracies-best-{}.png'.format(dataset_name))
    plt.close()


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing search_string and directory.
    """
    parser = argparse.ArgumentParser(description="Process and analyze files containing a specific string.")
    parser.add_argument(
        "--search-string",
        type=str,
        default="PROTEINS",
        help="String to search for in filenames."
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default="RESULTS-PROTEINS",
        help="Directory to search for files. Must be a subdir of the current directory."
    )
    parser.add_argument(
        "-k", "--kcorrelation",
        type=int,
        help="k of the k-correlation (up to, upper limit excuded)",
        default=5
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    search_string = args.search_string
    working_dir = os.path.join(os.getcwd(), args.directory)
    

    # Change to the specified directory so the cat and the grep can work later on
    print(working_dir)
    os.chdir(working_dir)
    files = os.listdir(working_dir)


    print(f"Searching for files containing '{search_string}' in directory '{working_dir}'.")

    # Find relevant dataset files
    DATASET_FILES = [file for file in files if search_string in file]


    data_to_plot = {}

    for filename in DATASET_FILES:
        print ("\nProcessing file: {}".format(filename))
        result = subprocess.run("cat {} | grep Test ".format(filename), shell=True, capture_output=True, text=True)
        text = result.stdout
        lines = text.strip().split('\n')
        try: 
            accuracies = [float(line.split()[-1]) for line in lines]
            # print('File: {}'.format(filename))
            # print the number of elements in the dataset
            result_len = subprocess.run(f"cat {filename} | grep Len | sort -u", shell=True, capture_output=True, text=True)
            print(result_len.stdout.strip())


            average = statistics.mean(accuracies)
            print('Average: {:.3f}'.format(average))

            variance = statistics.variance(accuracies)
            print('Variance: {:.3f}'.format(variance))

            print('Max:{:.3f}'.format(max(accuracies)) )
            print('Runs: {}'.format(len(accuracies)))
        
            
            data_to_plot[filename] = (average, variance, max(accuracies))    

        except Exception as e:
            print(e)


    # first_plot(data_to_plot, dataset_name=search_string, k=args.kcorrelation)
    # second_plot(data_to_plot, dataset_name=search_string, k=args.kcorrelation)
    # third_plot(data_to_plot, dataset_name=search_string, k=args.kcorrelation)
    #only_max_lines(data_to_plot, dataset_name=search_string, k=args.kcorrelation)