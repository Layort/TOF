# LTOS
Layout-controllable Text-Object Synthesis via Adaptive Cross-attention Fusions

## training

run `  CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=22411 main.py  --yaml_file=configs/text_object_control.yaml  --DATA_ROOT=./DATA   --batch_size=8  --name=cross_fusion_model  --official_ckpt_name="checkpoint_generation_text.pth"  `

--name: saved filename
--official_ckpt_name:  pretrained model (you should download the GLIGEN checkpoint from (https://huggingface.co/gligen/gligen-generation-text-box/blob/main/diffusion_pytorch_model.bin))

## inference

run ` python inference.py  ` 


## dataset_contruction 
in `data_construction/README.md`