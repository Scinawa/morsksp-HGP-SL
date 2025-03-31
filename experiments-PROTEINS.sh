#!/bin/bash


kcorre=4
count=5
B=35
DATASET="PROTEINS"


# Create the results directory if it doesn't exist
mkdir -p RESULTS-$DATASET
# Compress existing .txt files into a tar.gz archive named with the current date
current_date=$(date +%Y-%m-%d)
tar -czf RESULTS-$DATASET/results-$current_date.tar.gz RESULTS-$DATASET/*.txt

# Remove all .txt files after compression
rm RESULTS-$DATASET/*.txt


# Combined loops for SO and MO with different models and pooling ratios
for i in $(seq $count); do
    for correlation in $(seq 2 $kcorre); do
        echo "Seed $i for k=${correlation}"
        ###### MODEL 0
        # Model0 - SO with PR=0.5
        python3 extended-main.py --a 3 --b $B  --correlation ${correlation} --dataset $DATASET --read_dataset --pooling_ratio 0.5 --seed $i >> RESULTS-$DATASET/$DATASET-M0-PR05-SO-k${correlation}.txt

        # Model0 - MO with PR=0.5
        python3 extended-main.py --a 3 --b $B  --correlation ${correlation} --multi_orbits TRUE --dataset $DATASET --read_dataset --pooling_ratio 0.5 --seed $i >> RESULTS-$DATASET/$DATASET-M0-PR05-MO-k${correlation}.txt

        # Model0 - SO with PR=0.2
        python3 extended-main.py --a 3 --b $B  --correlation ${correlation} --dataset $DATASET --pooling_ratio 0.2 --read_dataset  --seed $i >> RESULTS-$DATASET/$DATASET-M0-PR02-SO-k${correlation}.txt

        # Model0 - MO with PR=0.2
        python3 extended-main.py --a 3 --b $B  --correlation ${correlation} --multi_orbits TRUE --dataset $DATASET --pooling_ratio 0.2 --read_dataset  --seed $i >> RESULTS-$DATASET/$DATASET-M0-PR02-MO-k${correlation}.txt


        # Original Model0 with and without PR
        python3 main.py --a 3 --b $B  --seed $i --correlation ${correlation} --dataset $DATASET >> RESULTS-$DATASET/$DATASET-M0-PR05-original.txt
        python3 main.py --a 3 --b $B  --seed $i --correlation ${correlation} --dataset $DATASET --pooling_ratio 0.2 >> RESULTS-$DATASET/$DATASET-M0-PR02-original.txt
        
        

        # ###### MODEL 1
        # # Model1 - SO
        # python3 extended-main.py --a 3 --b $B  --correlation ${correlation} --dataset $DATASET --model 1 --read_dataset  --seed $i >> RESULTS-$DATASET/$DATASET-M1-SO-k${correlation}.txt

        # # Model1 - MO
        # python3 extended-main.py --a 3 --b $B  --correlation ${correlation} --multi_orbits TRUE --dataset $DATASET --model 1 --read_dataset  --seed $i >> RESULTS-$DATASET/$DATASET-M1-MO-k${correlation}.txt

        # # Model1 - SO with PR
        # python3 extended-main.py --a 3 --b $B  --correlation ${correlation} --dataset $DATASET --model 1 --pooling_ratio 0.2 --read_dataset  --seed $i >> RESULTS-$DATASET/$DATASET-M1-PR-SO-k${correlation}.txt

        # # Model1 - MO with PR
        # python3 extended-main.py --a 3 --b $B  --correlation ${correlation} --multi_orbits TRUE --dataset $DATASET --model 1 --pooling_ratio 0.2 --read_dataset  --seed $i >> RESULTS-$DATASET/$DATASET-M1-PR-MO-k${correlation}.txt

        # Original Model1 with and without PR
        #python3 main.py --a 3 --b $B  --seed $i --correlation ${correlation} --dataset $DATASET --model 1 >> RESULTS-$DATASET/$DATASET-M1-original.txt
        #python3 main.py --a 3 --b $B  --seed $i --correlation ${correlation} --dataset $DATASET --model 1 --pooling_ratio 0.2 >> RESULTS-$DATASET/$DATASET-M1-PR-original.txt


        echo "Completed iteration $i for k=${correlation}"
    done
done
