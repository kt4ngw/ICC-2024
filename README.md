## Energy-Efficient Client Sampling for Federated Learning in Heterogeneous Mobile Edge Computing Networks

The paper is accepted (Proc. IEEE ICC).

**Title:** Energy-Efficient Client Sampling for Federated Learning in Heterogeneous Mobile Edge Computing Networks

**Author:**  Jian Tang; Xiuhua Li; Hui Li; Min Xiong; Xiaofei Wang; Victor C. M. Leung

---

First time writing code, inevitably not good. Thanks for your understanding.

### 1. Architecture

```
- src
  - alogorithms # sampling alogorithms
  - models # CNN Model
  - optimizers 
  - trainers # server in FL
  - utils
  - client.py # client in FL
  - cost.py
- args.py
- getdata.py # data processing
- main.py # main function
```



### 2. How to run

```
python main.py 
or
python main.py --algorithm propose
```

|    parameters    |                 explanations                  |
| :--------------: | :-------------------------------------------: |
|     --is_iid     |           data distribution is iid.           |
|  --dataset_name  |               name of dataset.                |
|   --round_num    |    number of round in communication round.    |
| --num_of_clients |             numer of the clients.             |
|   --c_fraction   | Proportion of clients selected in each round. |
|  --local_epoch   |       local train epoch of each client.       |
|   --algorithm    |             each sampling method.             |
|   --dirichlet    |   Delineate the Distribution of Dirichlet.    |
|       ...        |                                               |

### 3. citation

Finally, I would like to say this.

If this code was helpful for you, could you please cite this paper and give a star to this project?
I really appreciate that !!!

```
@INPROCEEDINGS{10623087,
  author={Tang, Jian and Li, Xiuhua and Li, Hui and Xiong, Min and Wang, Xiaofei and Leung, Victor C. M.},
  booktitle={Proc. IEEE ICC}, 
  title={Energy-Efficient Client Sampling for Federated Learning in Heterogeneous Mobile Edge Computing Networks}, 
  year={2024},
  pages={956-961},
  month={Jun.}}
```