# FL-DP
environment python3.8

Determine if GPU acceleration is supported and can run 

python3 cuda_test.py

#Usage

bash setup.sh

bash fl.sh

when --personal=0, this algorithm is fedavg

when --personal=1 and --ptype='privatefl', this algorithm is privatefl（usenix23）

when --personal=1 and --ptype='pd-ldpfl', this algorithm is PD-LDPFL

when --personal=1 and --ptype='pd-ldpfl++', this algorithm is PD-LDPFL++

when --personal=1 and --ptype='fedper', this algorithm is Fedper

Pending tasks：
Adding fedproto
