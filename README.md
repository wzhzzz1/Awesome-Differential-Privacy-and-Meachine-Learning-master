# FL-DP
environment python3.8

判断是否支持GPU加速可以运行 python3 cuda_test.py

#Usage

bash setup.sh

bash fl.sh

when --personal=0, this algorithm is fedavg

when --personal=0 and --ptype='single', this algorithm is privatefl

when --personal=0 and --ptype='double', this algorithm is PD-LDPFL

