import torch
from transformers.trainer import Trainer
from torch.optim import AdamW
import json
import os
import torch
import torch.nn.functional as F


def extract_image_parts(input_ids, hidden_states,labels):
    batch_size, max_length = input_ids.size()
    device = input_ids.device
    hidden_size = hidden_states[-1].size(-1) 
  
    # 1. Find the positions of token 8197 and token 8196 in each sample.
    start_token_id = 8197
    end_token_id = 8196

    # Create a mask for the start and end tokens.
    start_mask = (input_ids == start_token_id)  # (batch_size, max_length)
    end_mask = (input_ids == end_token_id)      # (batch_size, max_length)
    
    
    # Obtain the image start position and handle cases with no image or multiple images.
    start_positions=torch.nonzero(start_mask) #(image_num,2)
    img_cnt=start_positions.shape[0]

    if img_cnt==0:
        return None,None
    image_ids=[]
    image_hidden_states=[]
    for start_position in start_positions:
        batch_id=start_position[0]
        boi_pos=start_position[1]
        if labels[batch_id][boi_pos]==-100: 
            continue
        seq_length = 1024
        eoi_pos=boi_pos+seq_length+1

        image_ids.append(input_ids[batch_id][boi_pos+1:eoi_pos]) 
        image_hidden_states.append(hidden_states[-1][batch_id][boi_pos+1:eoi_pos]) 
      
    if len(image_ids)==0:
        return None,None
    image_ids=torch.stack(image_ids, dim=0)#(img_cnt,1024)
    image_hidden_states=torch.stack(image_hidden_states)#(img_cnt,1024,hidden_size)
    return image_ids, image_hidden_states


class AnoleTrainer(Trainer):
    def __init__(self, loss_net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_net = loss_net
        self.loss_net.to(self.args.device)
            
    def compute_loss(self, model, inputs, return_outputs=False):
        # ChameleonForCausalLM
        #print("input length", print(len(inputs["input_ids"])))
        outputs = model(input_ids=inputs["input_ids"], labels=inputs["labels"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
        hidden_states = outputs.hidden_states
      
        input_ids = inputs["input_ids"]
     
        assert hidden_states[-1].shape[1] == input_ids.shape[1]
        # cross entropy loss (local loss)
        local_loss = outputs.loss
        # global loss
        image_ids, image_hidden_states = extract_image_parts(input_ids=input_ids, hidden_states=hidden_states,labels=inputs["labels"])
        # image_ids = image_ids.to(input_ids)
        if image_ids!=None:
            global_loss = self.loss_net(input_ids=image_ids, generated_hidden_states=image_hidden_states)
        else:
            global_loss=0
        # total loss
        total_loss = local_loss + 1 * global_loss # global weight
      
        # total_loss = global_loss
        print(f"total loss: {total_loss}, local loss: {local_loss}, global loss: {global_loss}")
        return (total_loss, outputs) if return_outputs else total_loss
    
    