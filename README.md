# FL-DP
environment python3.8

Determine if GPU acceleration is supported and can run 

python3 cuda_test.py

#Usage

bash setup.sh

bash fl.sh

when --personal=0, this algorithm is fedavg

when --personal=1 and --ptype='single', this algorithm is privatefl

when --personal=1 and --ptype='double', this algorithm is PD-LDPFL

