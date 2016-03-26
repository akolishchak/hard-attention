--
--  action_policy.lua
--  hard-attention
--
--  Created by Andrey Kolishchak on 09/27/15.
--
require 'cunn'

action_policy = {}

local baseline_model = nn.Add(1)
baseline_model.bias:fill(0)
local basefile_criterion = nn.MSECriterion()
local zero_input = torch.Tensor()

function action_policy.init(opt)
  
  if opt.gpu > 0 then
    baseline_model:cuda()
    basefile_criterion:cuda()
    zero_input = zero_input:cuda()
  end
  zero_input:resize(opt.batch_size, 1):zero()
end

function action_policy.get_reward_loss(input, target)
  
  local max_val, max_index = torch.max(input, 2)
  local reward_loss = max_index:eq(target):cmul(max_val:exp())
  
  -- update baseline model
  local baseline = baseline_model:forward(zero_input)
  local loss = basefile_criterion:forward(baseline, reward_loss)
  
  baseline_model:zeroGradParameters()
  local dloss = basefile_criterion:backward(baseline, reward_loss)
  baseline_model:backward(zero_input, dloss)
  baseline_model:updateParameters(1e-3)
  
  -- generate new reward
  baseline = baseline_model:forward(zero_input)
  reward_loss:add(-1, baseline):mul(-1e-3)
    
  return reward_loss
end

return action_policy

