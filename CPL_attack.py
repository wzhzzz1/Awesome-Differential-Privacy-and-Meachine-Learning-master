
import time
get_ipython().run_line_magic('matplotlib', 'inline')
from pytorch_msssim import ssim
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
import torch.nn.functional as func
'''
def deep_leakage_from_gradients(model, origin_grad):
    dummy_data = torch.randn(origin_data.size())
    dummy_label = torch.randn(dummy_label.size())
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    for iters in range(300):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = criterion(dummy_pred, dummy_label)
            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = sum(((dummy_grad - origin_grad) ** 2).sum() \
                            for dummy_g, origin_g in zip(dummy_grad, origin_grad))

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

    return dummy_data, dummy_label
'''
tt = transforms.ToPILImage()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print (ssim(0.43*torch.unsqueeze(gt_data[0],dim=0),torch.unsqueeze(gt_data[0],dim=0),data_range=0).item())  #计算图像相似度，值在0到1之间，越接近1代表越相似
#print (torch.dist(0.6*torch.unsqueeze(gt_data[0],dim=0),torch.unsqueeze(gt_data[0],dim=0),2).item())  #计算图像距离
criterion = nn.CrossEntropyLoss()

for item in range(1):
    start = time.time()
    for rd in range(1):

        torch.manual_seed(200*rd)
        #dummy_data = torch.unsqueeze(torch.randn(gt_data[item].size()),0).to(device).requires_grad_(True)
        
        #dummy_data = torch.unsqueeze(torch.zeros(gt_data[item].size()),0).to(device).requires_grad_(True)
        #dummy_data = torch.unsqueeze(torch.ones(gt_data[item].size()),0).to(device).requires_grad_(True)

        
        #background = torch.unsqueeze(torch.zeros(gt_data[item].size()),0)
        #background[0,0,::] = 1
        #dummy_data = background.to(device).requires_grad_(True)
        ##dummy_data = (torch.unsqueeze(torch.randn(gt_data[item].size()),0)+background).to(device).requires_grad_(True)
        
        #surrogate = torch.unsqueeze(gt_data[item+1],0)
        #aaa = torch.rand([3,16,16])
        #surrogate[0,:,8:24,8:24] =aaa
        #dummy_data = surrogate.to(device).requires_grad_(True)    
        
        #dummy_data = torch.unsqueeze(gt_data[item+1],0).to(device).requires_grad_(True)
        
        #k = np.random.randint(0,95)
        #dummy_data = torch.unsqueeze(gt_data[k],0).to(device).requires_grad_(True)
        
        
        pat_1 = torch.rand([3,14,14])
        pat_2 = torch.cat((pat_1,pat_1),dim=1)
        pat_4 = torch.cat((pat_2,pat_2),dim=2)
        dummy_data = torch.unsqueeze(pat_4,dim=0).to(device).requires_grad_(True)
        
        
        #aaa = torch.rand([3,8,8])
        #bbb = torch.cat((aaa,aaa),dim=1)
        #ccc = torch.cat((bbb,bbb),dim=1)
        #ddd = torch.cat((ccc,ccc),dim=2)
        #eee = torch.cat((ddd,ddd),dim=2)
        #dummy_data = torch.unsqueeze(eee,dim=0).to(device).requires_grad_(True)
        
        #aaa = torch.rand([3,4,4])
        #bbb = torch.cat((aaa,aaa),dim=1)
        #ccc = torch.cat((bbb,bbb),dim=1)
        #ddd = torch.cat((ccc,ccc),dim=1)
        #eee = torch.cat((ddd,ddd),dim=2)
        #fff = torch.cat((eee,eee),dim=2)
        #ggg = torch.cat((fff,fff),dim=2)
        #dummy_data = torch.unsqueeze(ggg,dim=0).to(device).requires_grad_(True)
        
        
        #dummy_data = plt.imread("./attack_image/replacement_69.png")
        #print (dummy_data.shape)
        #dummy_data = torch.FloatTensor(dummy_data).to(device)
        #dummy_data = dummy_data.transpose(2,3).transpose(1,2)
        

        label_pred=torch.argmin(torch.sum(original_dy_dx[item][-2], dim=-1), 
                                dim=-1).detach().reshape((1,)).requires_grad_(False)
        #print (original_dy_dx[item][-1].shape)
        #print (original_dy_dx[item][-1].argmin())
        
        #print (torch.sum(original_dy_dx[item][-2], dim=-1).argmin())
        
        plt.imshow(tt(dummy_data[0].cpu()))
        plt.title("Dummy data")
        #plt.savefig("./random_seed/index_%s_rand_seed_%s_label_%s"%(item,rd,torch.argmax(dummy_label, dim=-1).item()))

        plt.clf()

        print("stolen label is %d." % label_pred.item())
        
        
        #optimizer = torch.optim.LBFGS([dummy_data,dummy_label])
        optimizer = torch.optim.LBFGS([dummy_data,])
        #optimizer = torch.optim.AdamW([dummy_data,],lr=0.01)
        #optimizer = torch.optim.SGD([dummy_data,],lr=0.01)
      
       

        history = []
        
        percept_dis = np.zeros(300)
        recover_dis = np.zeros(300)
        for iters in range(300):
            
          
            #percept_dis[iters]=ssim(dummy_data,torch.unsqueeze(gt_data[item],dim=0),data_range=0).item()
            #recover_dis[iters]=torch.dist(dummy_data,torch.unsqueeze(gt_data[item],dim=0),2).item()
           
            history.append(tt(dummy_data[0].cpu()))
            def closure():
                optimizer.zero_grad()

                pred = net(dummy_data) 
                
                #dummy_onehot_label = F.softmax(dummy_label, dim=-1).long()
                
                #dummy_loss = criterion(pred, dummy_onehot_label) # TODO: fix the gt_label to dummy_label in both code and slides.
                ##print (pred)
                ##print (label_pred)
            
                dummy_loss = criterion(pred, label_pred)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                ##dummy_dy_dp = torch.autograd.grad(dummy_loss, dummy_data, create_graph=True)
                ##print (dummy_dy_dp[0].shape)  

                grad_diff = 0
                grad_count = 0
                #count =0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx[item]): # TODO: fix the variablas here
                   
                    #if iters==500 or iters== 1200:
                    #print (gx[0])
                    #    print ('hahaha')
                    #print (gy[0])
                    lasso = torch.norm(dummy_data,p=1)
                    ridge = torch.norm(dummy_data,p=2)
                    grad_diff += ((gx - gy) ** 2).sum() #+ 0.0*lasso +0.01*ridge 
                    
                    #print (gx.shape)

                    grad_count += gx.nelement()
                

                    #if count == 9:
                    #    break
                    #count=count+1
                # grad_diff = grad_diff / grad_count * 1000
                
                #grad_diff += ((original_pred[item]-pred)**2).sum()
               
                
                
                
                grad_diff.backward()
                #print (count)

                #print (dummy_dy_dx)
                #print (original_dy_dx)


                return grad_diff



            optimizer.step(closure)
            if iters % 5 == 0: 
                current_loss = closure()
                #if iters == 0: 
                print ("%.8f" % current_loss.item())
                #print(iters, "%.8f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))

        
        
        #plt.figure(figsize=(18, 12))
        #for i in range(60):
        #  plt.subplot(6, 10, i + 1)
        #  plt.imshow(history[i * 5])
        #  plt.title("iter=%d" % (i * 5))
        #  plt.axis('off')
        
        plt.figure(figsize=(12, 1.5))
        #iter_idx = [0,20,40,60,80,100,120,140,160,180]
        plt.figure(figsize=(6.5, 1.2))
        #iter_idx = [0,1000,2000,3000,4000,5000]
        iter_idx = [0,5,10,20,50,100]
        

        for i in range(6):
          plt.subplot(1, 6, i + 1)
          plt.imshow(history[iter_idx[i]])
          plt.title("iter=%d" % (iter_idx[i]))
          plt.axis('off')
            
        #np.savetxt('ssim_random2',percept_dis,fmt="%4f")
        #np.savetxt('mse_random2',recover_dis,fmt="%4f")
        
        #print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())
        plt.savefig("./attack_image/index_%s_rand_%s_label_%s"%(item,rd, label_pred.item()))
        #plt.clf()
       
    duration = time.time()-start
    #print ("Running time is %.4f." %(duration/10.0) )
    print (duration/10.0 )