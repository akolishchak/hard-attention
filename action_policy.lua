--
--  action_policy.lua
--  hard-attention
--
--  Created by Andrey Kolishchak on 09/27/15.
--
require 'cunn'

action_policy = {}

local reward_criterion = nn.BCECriterion()

function action_policy.get_reward_loss(input, target)
  
  local max_val, max_index = torch.max(input, 2)
  local reward_loss = max_index:eq(target):cmul(max_val:exp())
  
  -- generate new reward
  local baseline = reward_loss:mean()
  reward_loss:add(-baseline):mul(-1e-3)
    
  return reward_loss
end

function action_policy.get_demo_reward_loss(input, target)
  
  local loss = reward_criterion:forward(input:ge(0), target:ge(0))
  local reward_loss = target.new(target:size(1)):fill(-loss)
  
  -- generate new reward
  local baseline = reward_loss:mean()
  reward_loss:add(-baseline):mul(-1e-3)
    
  return reward_loss
end


return action_policy

