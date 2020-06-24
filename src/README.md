- Before u run the main code, u need to download the csv file of the raw data, and place them in the directory ../public/raw_csv_data.

- I have copmuted the ground-truth-rank of the given raw-data in the directory ../public/pre_computed, so the main code will read the pre-computed file to speed up the evaluation.

- I have slightly modified the evaluation code from the original one.

## There are two options to run the code:

1. Option '0' for the given training-data whose size is 2468285
```bash
> sh run.sh 0
```
2. Option '1' for the given testing-data whose size is 200000
```bash
> sh run.sh 1
```
