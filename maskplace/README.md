# MaskPlace

Code taken from https://github.com/laiyao1/maskplace

## Installation

1. Navigate to the project directory:
    ```bash
    cd maskplace
    ```

2. Create a new conda environment using the provided environment.yaml file:
    ```bash
    conda env create -f environment.yaml
    ```

3. Activate the conda environment:
    ```bash
    conda activate maskplace
    ```

4. Run the main trainer:
    ```bash
    python PPO2.py
    ```


**Note:**

1. Ensure that `pytorch` is installed with CUDA (for faster execution).
   
2. After training (running PPO2.py), checkpoints are created in the `save_models` directory. Also the RL agent is trained on the dataset `adaptec`.

3. To run the `maskplace` from a checkpoint, change the location of the `load_model_path` variable in `PPO2.py` to the checkpoint path and run with the `--is_test` flag:

    ```bash
    python PPO2.py --is_test
    ```

4. To save figures while training, pass the `--save_fig` flag:

    ```bash
    python PPO2.py --save_fig
    ```

5. Also refer to the code in `PPO2.py` for running the placement script with different parameters.

6. The `place_db` file gets the macros from the dataset ('adaptec' and 'ariane') to be read by the algorithm.

7. Please note that we have changed the 'adaptec' dataset. For the original dataset, please refer to [https://github.com/laiyao1/maskplace](https://github.com/laiyao1/maskplace).
   
8. The `tb_log` directories are the tensorfboard log files. The experiment run can be changed by changing the `SummarWriter` path to the new experiment directory.
