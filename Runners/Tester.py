import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from math import log10

from FDL.Views_Params import DiscreteApertureFSParams, AllFocusParams
from Dataloaders.Dataloader import LightFieldDataset
from Utils.Light_Field_Utils import pad_lf
from Utils.Fourier_Utils import gen_frequency_grid
from torchvision.utils import save_image

'''
---------------------------------------------------------------
CLASS Tester
    Tester of a FDL pytorch model to reconstruct a light field
---------------------------------------------------------------
'''
class Tester():
    def __init__(self, config, testing_dataset, model_name, save_directory, output_type):
        super(Tester,self).__init__()
        self.config = config
        self.testing_dataset = testing_dataset
        self.model_name = model_name
        self.save_directory = save_directory
        self.output_type = output_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ''' Read a text file line by line ''' 
    def file_directory_list(self, file):
        file = open(file, 'r')
        lines = file.read().splitlines()
        return lines
    
    ''' Load the FDL optimization solver mentioned in the configuration file '''     
    def load_solver(self):
        print("\nLOADING SOLVER")
        print("USING",self.config["solver"]["name"],"SOLVER")
        if self.config["solver"]["name"] == "UnrolledADMMFDL":
            from Solvers.Unrolled_ADMM_FDL import UnrolledADMMFDL
            return UnrolledADMMFDL(nb_channels=self.config["data"]["nb_channels"], nb_iteration=self.config["solver"]["nb_iteration"], nb_fdl=self.config["task"]["nb_fdl"], rho_initial_val=self.config["solver"]["rho"], device=self.device)
        else:
            print("Unknown solver")
            import sys
            sys.exit()
    
    ''' Load a pre-trained model for the FDL optimization solver '''
    def load_model(self, solver):
        print("\nLOADING MODEL")
        if os.path.exists(os.path.join("Models/"+self.model_name+".ckpt")):
            print("USING PRETRAINED MODEL IN:",os.path.join("Models/"+self.model_name+".ckpt"))
            if self.device != 'cpu':
                saved_dict = torch.load(os.path.join("Models/"+self.model_name+".ckpt"))
            else:
                saved_dict = torch.load(os.path.join("Models/"+self.model_name+".ckpt"), map_location='cpu')
            solver.load_state_dict(saved_dict['solver_state_dict'])
        else:
            print("WARNING: USING RANDOM MODEL")  
            
        return solver
        
    ''' Load the testing light field dataset '''    
    def load_dataset(self, nb_views):
        print("\nLOADING DATASET")
        patch_transform = transforms.Compose([])

        testing_dataset = self.file_directory_list(self.testing_dataset)
        print("USING TESTING DATASETS IN:")
        for path in testing_dataset:
            print(path)

        testing_dataset = LightFieldDataset(directories=testing_dataset,transform=patch_transform, angular_dims=[nb_views,nb_views])
        testing_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=8)
        
        return testing_dataloader
        
    ''' prepare the solver, the input, and the groundtruth for the forward pass '''
    def prepare_inputs(self, solver, views_params, batch, u, v, pad):
        gt = pad_lf(batch[0], pad, mode='reflect', use_window=True).to(self.device)
        inputs = views_params.gen_views_from_light_field(gt, u, v)
        views_fft = torch.fft.fftn(inputs, dim=(-2, -1))
        n_freq_x = views_fft.shape[-1]
        n_freq_y = views_fft.shape[-2]
        solver.fdl_model.w_x, solver.fdl_model.w_y = gen_frequency_grid(n_freq_x, n_freq_y, half_x_dim=True, centered_zero=True)
        solver.fdl_model.to(self.device)
        fdl = torch.zeros(inputs.shape[:2] + solver.d_fdl.shape[-1:] + inputs.shape[-2:],device=inputs.device, dtype=inputs.dtype)
        
        return solver, gt, inputs, fdl
    
    ''' prepare the output and the groundtruth for the loss computation '''    
    def prepare_outputs(self, solver, fdl, gt, u, v, pad, d_fs_gt):
        if self.output_type == "views":
            render_params = AllFocusParams(u, v)
            output = solver.generate_views(fdl, render_params)
            output = output.unflatten(2, u.shape)    
            output = torch.fft.ifftn(output, dim=(-2, -1)).real
            output = output[..., pad[2]:output.shape[-2] - pad[3], pad[0]:output.shape[-1] - pad[1]]   
            gt = gt[..., pad[2]:gt.shape[-2] - pad[3], pad[0]:gt.shape[-1] - pad[1]]            
        elif self.output_type == "fs":
            render_params = DiscreteApertureFSParams(u, v, d_fs_gt)
            output = solver.generate_views(fdl, render_params)
            output = output.unsqueeze(2)
            output = torch.fft.ifftn(output, dim=(-2, -1)).real
            output = output[..., pad[2]:output.shape[-2] - pad[3], pad[0]:output.shape[-1] - pad[1]] 

            gt = render_params.gen_views_from_light_field(gt, u, v)
            gt = gt.unsqueeze(2)
            gt = gt[..., pad[2]:gt.shape[-2] - pad[3], pad[0]:gt.shape[-1] - pad[1]]
        else:
            print("Unknown output type")
            import sys
            sys.exit()

        return gt, output
    
    ''' set the rng seed to have consistent output '''
    def set_rng(self, seed=0):
        rng_cpu = torch.get_rng_state()
        rng_gpu = None
        if torch.cuda.is_available():
            rng_gpu = torch.cuda.get_rng_state_all()
        torch.manual_seed(seed)
        
        return rng_cpu, rng_gpu
        
    ''' reset a pre-set rng seed '''       
    def reset_rng(self, rng_cpu, rng_gpu):
        torch.set_rng_state(rng_cpu)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_gpu)
            
    ''' save reconstructed light field views '''       
    def save_results(self, lf_res, lf_name):
        lf_directory = os.path.join(self.save_directory,lf_name)
        if not os.path.isdir(self.save_directory):
            os.mkdir(self.save_directory)
        if not os.path.isdir(lf_directory):
            os.mkdir(lf_directory)
        print("Saving reconstructed views in", lf_directory)
            
        for v in range(lf_res.shape[2]):
            for u in range(lf_res.shape[3]):
                save_image(lf_res[0,:,v,u,:,:],os.path.join(lf_directory,"lf_"+str(v)+"_"+str(u)+".png"))
    
    ''' Compute the testing of a FDL optimization model '''
    def test(self):
        torch.cuda.empty_cache()
        ################# PARAMETERS #################
        nb_views = self.config["data"]["angular_dims"]
        nb_measurements = self.config["data"]["number_of_shots"]
        v, u = torch.meshgrid(torch.arange(float(int(nb_views/2.0)), -float((int(nb_views/2.0)+1)), -1), torch.arange(-float(int(nb_views/2.0)), float(int(nb_views/2.0)+1)))
        v = v.to(self.device)
        u = u.to(self.device)
        d_fs = torch.tensor(self.config["task"]["disparity_focal_stack"])
        d_fs_gt = torch.linspace(-1, 1, steps=11).to(self.device)
        pad = (8, 8, 8, 8)

        views_params = DiscreteApertureFSParams(u, v, d_fs)
        
        ################# LOAD SOLVER #################
        solver = self.load_solver().to(self.device)
        
        ################# LOAD MODEL #################
        solver = self.load_model(solver)
        
        ################# LOAD OPTIMIZER #################        
        loss_function = torch.nn.MSELoss()
        
        ################# LOAD DATASET #################
        testing_dataloader = self.load_dataset(nb_views)
        
        ################# START TRAINING #################
        print("\nSTARTING TESTING")
        best_average_error = -1
        test_results = []
        
        nb_batch = 1.0*testing_dataloader.__len__()
        patch_done = 0
        
        ################# TESTING PHASE #################        
        rng_cpu, rng_gpu = self.set_rng()

        for ii, batch in enumerate(testing_dataloader):
            with torch.no_grad():
                lf_name = batch[1][0].split('/')[-1]
                
                solver, gt, inputs, fdl = self.prepare_inputs(solver, views_params, batch, u, v, pad)
                output = solver(fdl, inputs, views_params)
                gt, output = self.prepare_outputs(solver, output, gt, u, v, pad, d_fs_gt)

                loss_value = loss_function(output, gt) 
                if loss_value != 0:
                    psnr = 10 * log10(1/loss_value.item())
                else:
                    psnr = 0
                test_results.append(psnr)
            
            patch_done += 1
            print("[" + str(int(patch_done)) + "/" + str(int(nb_batch)) + "]","- PSNR:",str(round(psnr,2)) + "db","- Mean PSNR:", str(round(np.mean(test_results),2)) + "db","- Memory:", str(torch.cuda.max_memory_allocated(self.device)/(10**9)) + "Gb")
            
            if self.save_directory is not None:
                self.save_results(output[:,:,:,:,0:-1,:], lf_name)
         
        self.reset_rng(rng_cpu, rng_gpu)
            
        ave_error_test = sum(test_results) / len(test_results)
        
        print()
        print("Average testing PSNRs:",ave_error_test, "db")
        print()
    
    ''' Launch the testing '''
    def forward(self):
        self.test()