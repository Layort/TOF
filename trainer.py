import cv2
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import numpy as np
import random
import time 
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
import shutil
import torchvision
from convert_ckpt import add_additional_channels
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from distributed import get_rank, synchronize, get_world_size
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from copy import deepcopy
from inpaint_mask_func import draw_masks_from_boxes
from ldm.modules.attention import BasicTransformerBlock
# from cldm.recognizer import adjust_image, create_predictor, crop_image, pre_process
from ldm.modules.diffusionmodules.util import  extract_into_tensor
import pdb
import time
try:
    from apex import amp
except:
    pass  
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #
save_id = 0
device = 'cuda'
class ImageCaptionSaver:
    def __init__(self, base_path, nrow=8, normalize=True, scale_each=True, range=(-1,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self, images, real, mask_image, masked_real, captions, seen):
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'.png')
        torchvision.utils.save_image( images, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range )
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_real.png')
        torchvision.utils.save_image( real, save_path, nrow=self.nrow)

        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_mask.png')
        torchvision.utils.save_image( mask_image, save_path, nrow=self.nrow)
        
        if masked_real is not None:
            # only inpaiting mode case 
            save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_mased_real.png')
            torchvision.utils.save_image( masked_real, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range)

        assert images.shape[0] == len(captions)

        save_path = os.path.join(self.base_path, 'captions.txt')
        with open(save_path, "a") as f:
            f.write( str(seen).zfill(8) + ':\n' )    
            for cap in captions:
                f.write( cap + '\n' )  
            f.write( '\n' ) 

def load_ckpt(ckpt_path):
    
    saved_ckpt = torch.load(ckpt_path, map_location="cpu")
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    # model.load_state_dict( saved_ckpt['model'] )
    # autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    # text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    # diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config



def read_official_ckpt(ckpt_path):      
    "Read offical pretrained SD ckpt and convert into my style" 
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}
    if get_rank()==0:
        print(state_dict.keys())
 
    
    for k,v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            out["model"][k.replace("model.diffusion_model.", "")] = v 
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v 
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v 
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v  
        else:
            out["diffusion"][k] = v     
    return out 


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def sub_batch(batch, num=1):
    # choose first num in given batch 
    num = num if num > 1 else 1 
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch


def wrap_loader(loader):
    while True:
        for batch in loader:  # TODO: it seems each time you have the same order for all epoch?? 
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    if get_rank()==0:
        print("total_trainable_params_count is: ", total_trainable_params_count)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

           
def create_expt_folder_with_auto_resuming(OUTPUT_ROOT, name):
    name = os.path.join( OUTPUT_ROOT, name )
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join( name, previous_tag, 'checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming ckpt found '+ potential_ckpt)
                break 
        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 

class Trainer:
    def __init__(self, config):

        self.config = config
        self.device = torch.device("cuda")

        self.l_simple_weight = 1
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml")  )
            self.config_dict = vars(config)
            torch.save(  self.config_dict,  os.path.join(self.name, "config_dict.pth")     )


        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        self.model = instantiate_from_config(config.model).to(self.device)
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
        self.diffusion = instantiate_from_config(config.diffusion).to(self.device)

        
        #####  load gligen_model  pretrained_weights#####
 
        state_dict = torch.load(os.path.join('./gligen_checkpoints',config.official_ckpt_name), map_location="cpu")
        pretrained_weights =  state_dict['model']
        # print('pretrained_weights',pretrained_weights.keys())


        original_params_names = list( state_dict["model"].keys()  ) # used for sanity check later 
        print('original_params_names',len(original_params_names))
        
        scratch_dict = self.model.state_dict() 
        print('scratch_dict', scratch_dict.keys())
        print('scratch_dict',len(scratch_dict.keys()))
        
        ## load orc model
        # rec_model_dir = "./ocr_weights/ppv3_rec_en.pth"
        # self.ocr_predictor = create_predictor(rec_model_dir,model_lang='en').cuda()
        
        target_control_dict = {}
        k_new_name = 0
        k_ori_params = 0
        k_zero_params = 0

        #initial control-model
        for k in scratch_dict.keys():
            the_first_str = k.split('.')[0]
            params_name = k.replace('control_model.',"") 
            if the_first_str=="control_model":
                if params_name in pretrained_weights.keys():
                    k_new_name+=1
                    target_control_dict[k] = pretrained_weights[params_name].clone()
                else :
                    # print('not know k**********',k)
                    k_zero_params+=1
                    target_control_dict[k] = scratch_dict[k].clone()
            elif  "control_alpha_attn_list" in k:
                k_zero_params+=1  
                # print('control_alpha_attn*****k******',k)
                target_control_dict[k] =  scratch_dict[k].clone()
        #686  #298 middle
            else:
                if 'attention_' in k:
                    k_zero_params+=1
                    if get_rank()==0:
                        print('attention_****k***',k)
                    target_control_dict[k] = scratch_dict[k].clone()
                else:
                    # print('k_control',k)
                    k_ori_params+=1
                    target_control_dict[k] = pretrained_weights[k].clone()
                   
        print('len(k_new_name)',k_new_name)  
        print('len(zero_params)',k_zero_params) 
        print('len(k_ori_params)',k_ori_params) 
        print('len(target_control_dict)',len(target_control_dict)) 
    
        self.model.load_state_dict(target_control_dict)

        self.autoencoder.load_state_dict( state_dict["autoencoder"]  )
        self.text_encoder.load_state_dict( state_dict["text_encoder"]  )
        self.diffusion.load_state_dict( state_dict["diffusion"]  )
        
        self.autoencoder.eval()
        self.text_encoder.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)
        disable_grads(self.model)  
        # disable_grads(self.ocr_predictor)
        for p in self.model.control_model.parameters():
            p.requires_grad = True

        if self.config.ckpt is not None:
            first_stage_ckpt = torch.load(self.config.ckpt, map_location="cpu")
            self.model.load_state_dict(first_stage_ckpt["model"])
        

        # = = = = = = = = = = = = = = = = = create opt = = = = = = = = = = = = = = = = = #
        
        #********  controlnet params should be trained   ********* 
        params = []
        trainable_names = []
        all_params_name = []
        # print('type(self.control_mode)',type(self.control_model))
        # print('type(self.model)',type(self.model))
        for name, p in self.model.named_parameters():
            if ("control_model" in name  ):
                params.append(p) 
                trainable_names.append(name)
            elif 'control_alpha_attn_list' in  name:
                params.append(p) 
                trainable_names.append(name)
            elif "cross_attention_list" in name:
                params.append(p) 
                trainable_names.append(name)
            elif "attention_middle" in name:
                params.append(p) 
                trainable_names.append(name)
            all_params_name.append(name)
            

        if get_rank()==0:
            print('len(trainable_names)',len(trainable_names))
            print('trainable_names',trainable_names)
        
 
        # print('trainable_names',trainable_names)
        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay) 
        count_params(params)
    
       

        #  = = = = = EMA... It is worse than normal model in early experiments, thus never enabled later = = = = = = = = = #
        if config.enable_ema:
            self.master_params = list(self.model.parameters()) 
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()


        # = = = = = = = = = = = = = = = = = = = = create scheduler = = = = = = = = = = = = = = = = = = = = #
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_iters)
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps)
        else:
            assert False 

        
        # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #  
        train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None
        #add glyph-image
        print(config.train_dataset_names, config.DATA_ROOT)
        dataset_train = ConCatDataset(config.train_dataset_names, config.DATA_ROOT, train=True, repeats=train_dataset_repeats)
        sampler = DistributedSampler(dataset_train, seed=config.seed) if config.distributed else None 
        print(config.batch_size, (sampler is None) , sampler)
        loader_train = DataLoader( dataset_train,  batch_size=config.batch_size, 
                                                   shuffle=(sampler is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True, 
                                                   sampler=sampler)
        self.dataset_train = dataset_train
        self.loader_train = wrap_loader(loader_train)

        if get_rank() == 0:
            total_image = dataset_train.total_images()
            print("Total training images: ", total_image)     
        


      
        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0  
        if checkpoint is not None:
            print('= = = = = = = = load from autoresuming ckpt = = = = = = ')
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            if config.enable_ema:
                self.ema.load_state_dict(checkpoint["ema"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.starting_iter = checkpoint["iters"]
            if self.starting_iter >= config.total_iters:
                synchronize()
                print("Training finished. Start exiting")
                exit()


        # = = = = = = = = = = = = = = = = = = = = misc and ddp = = = = = = = = = = = = = = = = = = = =#    
        
        # func return input for grounding tokenizer 
        self.grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
        self.model.grounding_tokenizer_input = self.grounding_tokenizer_input
        
        # func return input for grounding downsampler  
        self.grounding_downsampler_input = None
        if 'grounding_downsampler_input' in config:
            self.grounding_downsampler_input = instantiate_from_config(config.grounding_downsampler_input)

        if get_rank() == 0:       
            self.image_caption_saver = ImageCaptionSaver(self.name)

        if config.distributed:
            self.model = DDP( self.model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False )
            # self.control_model = DDP( self.control_model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False,find_unused_parameters=True )
    
    @torch.no_grad()
    def get_input(self, batch):
        z = self.autoencoder.encode( batch["image"] )
        # print(z.shape) [B,4,64,64]

        # *******  glyph_image should be added  ****
        hint =  batch["text_mask_image_tensor"]
        context = self.text_encoder.encode( batch["caption"]  )
        
        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999
        
        inpainting_extra_input = None
        if self.config.inpaint_mode:
            # extra input for the inpainting model 
            inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda()
            masked_z = z * inpainting_mask
            inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
        
        grounding_extra_input = None
        if self.grounding_downsampler_input != None:
            grounding_extra_input = self.grounding_downsampler_input.prepare(batch)

        return z, hint, t, context, inpainting_extra_input, grounding_extra_input 


    def run_one_step(self, batch):
        global save_id
        # added hint
        t1 = time.time()
        x_start, hint, t, context, inpainting_extra_input, grounding_extra_input = self.get_input(batch)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)
    

        grounding_input = self.grounding_tokenizer_input.prepare(batch)
        input = dict(x=x_noisy, 
                    timesteps=t, 
                    context=context, 
                    hint = hint,
                    inpainting_extra_input=inpainting_extra_input,
                    grounding_extra_input=grounding_extra_input,
                    grounding_input=grounding_input)

        model_output = self.model(input)


        loss = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight
        # calculate orc loss
        # t2 = time.time()
        # orc_loss = 0
        # batch_here =  len(batch["id"])
        # model_wo_wrapper = self.model.module if self.config.distributed else self.model
        # plms_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)
        # shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)
        # uc = self.text_encoder.encode( batch_here*[""] )
        # t3 = time.time()
        # gen_latent_imgs = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=5)
        # gen_imgs = self.autoencoder.decode(gen_latent_imgs)
        # true_imgs = batch["image"]
        # t4 = time.time()
        # # phis = extract_into_tensor(self.diffusion.alphas_cumprod, t, true_imgs.shape)
        # phis = self.diffusion.alphas_cumprod.gather(-1,t)
        # for true_img,gen_img,img_text_boxes,img_text_mask,phi in zip(true_imgs,gen_imgs,batch["text_bboxes"],batch["text_boexes_masks"],phis):
        #     true_img = (true_img+1)*127.5
        #     for bbox, mask in zip(img_text_boxes,img_text_mask):
        #         if(mask):
        #             bbox *= 512
        #             bbox= np.array(bbox.permute(1,0).cpu())
        #             gen_img = torch.clamp(gen_img, min=-1, max=1) * 0.5 + 0.5
        #             gen_img = gen_img* 255 
        #             pred_text = adjust_image(bbox, gen_img)
        #             true_text = adjust_image(bbox, true_img)
        #             print("pred_text_shape:",pred_text.shape,"true_text_shape:",true_text.shape)
        #             if(pred_text.shape[1] <= 16):
        #                 continue
        #             cv2.imwrite(f"Middle_output/{save_id}_0.jpg",np.array(true_text.permute(1,2,0).cpu()).astype(np.uint8))
        #             cv2.imwrite(f"Middle_output/{save_id}_1.jpg",np.array(pred_text.permute(1,2,0).cpu()).astype(np.uint8))
        #             save_id += 1
        #             pred_out = self.ocr_predictor(pred_text.unsqueeze(0))
        #             true_out = self.ocr_predictor(true_text.unsqueeze(0))
        #             orc_loss += phi*torch.nn.functional.mse_loss(pred_out["ctc_neck"],true_out["ctc_neck"])/(gen_img.shape[1]*gen_img.shape[2])
        #             # print("true_img.shape",true_img.shape,"gen_img.shape",gen_img.shape)               
        #             print("orc_loss",orc_loss)
        #             print("loss",loss)
        # t5 = time.time()
        # print(t2-t1,t3-t2,t4-t3,t5-t4)
        # loss_mask = torch.nn.functional.mse_loss(model_output*mask_image, noise*mask_image)*self.large_weight
        # self.loss_dict = {"loss": loss.item() + 0.5 * orc_loss.item()}
        self.loss_dict = {"loss": loss.item()}

        print('loss',self.loss_dict)
        return loss 
        
  

        
    def start_training(self):
        # last_loss = 1

        iterator = tqdm(range(self.starting_iter, self.config.total_iters), desc='Training progress',  disable=get_rank() != 0 )
        self.model.train()
        for iter_idx in iterator: # note: iter_idx is not from 0 if resume training
            self.iter_idx = iter_idx

            self.opt.zero_grad()
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            loss = self.run_one_step(batch)
            print(loss)
            loss.backward()
            
            self.opt.step() 
            self.scheduler.step()
            if self.config.enable_ema:
                update_ema(self.ema_params, self.master_params, self.config.ema_rate)

            # ********** debug *********
            if (get_rank() == 0):
                if (iter_idx % 10 == 0):
                    self.log_loss() 
                if (iter_idx == 0)  or  ( iter_idx % self.config.save_every_iters == 0 )  or  (iter_idx == self.config.total_iters-1):
                    self.save_ckpt_and_result()
            synchronize()

        
        synchronize()
        print("Training finished. Start exiting")
        exit()


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )  # we add 1 as the actual name
    

    @torch.no_grad()
    def save_ckpt_and_result(self):

        model_wo_wrapper = self.model.module if self.config.distributed else self.model
        

        iter_name = self.iter_idx + 1     # we add 1 as the actual name

        if not self.config.disable_inference_in_training:
            # Do an inference on one training batch 
            batch_here = self.config.batch_size
            batch = sub_batch( next(self.loader_train), batch_here)
            batch_to_device(batch, self.device)

            
            if "boxes" in batch:
                real_images_with_box_drawing = [] # we save this durining trianing for better visualization
                for i in range(batch_here):
                    temp_data = {"image": batch["image"][i], "boxes":batch["boxes"][i],"text_boxes": batch["text_boxes"][i]}
                    im = self.dataset_train.datasets[0].vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)  # image mask_image mask
                    real_images_with_box_drawing.append(im)
                real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)
            else:
                real_images_with_box_drawing = batch["image"]*0.5 + 0.5 
                
            
            uc = self.text_encoder.encode( batch_here*[""] )
            context = self.text_encoder.encode(  batch["caption"]  )
            
            hint = batch["text_mask_image_tensor"] 

         
            plms_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)      
            shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)
            
            inpainting_extra_input = None
            if self.config.inpaint_mode:
                z = self.autoencoder.encode( batch["image"] )
                inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda()
                masked_z = z*inpainting_mask
                inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)
            
            grounding_extra_input = None
            if self.grounding_downsampler_input != None:
                grounding_extra_input = self.grounding_downsampler_input.prepare(batch)
            
            grounding_input = self.grounding_tokenizer_input.prepare(batch)
            input = dict( x=None, 
                          timesteps=None, 
                          context=context, 
                          hint=hint,
                          inpainting_extra_input=inpainting_extra_input,
                          grounding_extra_input=grounding_extra_input,
                          grounding_input=grounding_input )
            samples = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=5)
            
            autoencoder_wo_wrapper = self.autoencoder 
            samples = autoencoder_wo_wrapper.decode(samples).cpu()
            samples = torch.clamp(samples, min=-1, max=1)

            masked_real_image =  batch["image"]*torch.nn.functional.interpolate(inpainting_mask, size=(512, 512)) if self.config.inpaint_mode else None
            self.image_caption_saver(samples, real_images_with_box_drawing, hint,  masked_real_image, batch["caption"], iter_name)

        ckpt = dict(model = model_wo_wrapper.state_dict(),
                    text_encoder = self.text_encoder.state_dict(),
                    autoencoder = self.autoencoder.state_dict(),
                    diffusion = self.diffusion.state_dict(),
                    opt = self.opt.state_dict(),
                    scheduler= self.scheduler.state_dict(),
                    iters = self.iter_idx+1,
                    config_dict=self.config_dict,
        )
        if self.config.enable_ema:
            ckpt["ema"] = self.ema.state_dict()
        
         
        if (iter_name-1) % (5 * self.config.save_every_iters) == 0 :
            torch.save( ckpt, os.path.join(self.name, "checkpoint_"+str(iter_name).zfill(8)+".pth") )
            torch.save( ckpt, os.path.join(self.name, "checkpoint_latest.pth") )


# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=22411 main.py  --yaml_file=configs/text_object_control.yaml  --DATA_ROOT=./DATA   --batch_size=8  --name=cross_fusion_model  --official_ckpt_name="checkpoint_generation_text.pth"
