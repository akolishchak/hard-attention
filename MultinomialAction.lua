--
--  MultinomialAction.lua
--  hard-attention
--
--  Created by Andrey Kolishchak on 09/27/15.
--

require 'nn'
local MultinomialAction, parent = torch.class('nn.MultinomialAction', 'nn.Module')

function MultinomialAction:__init()
  parent.__init(self)
  
  self.epsilon = 1e-12
  self.input = torch.Tensor()
end
 
 
function MultinomialAction:updateOutput(input)  
  
  self.input:resizeAs(input):copy(input):add(self.epsilon)
  
  if self.train == true then
    self.output = torch.multinomial(self.input, 1)
  else
    _, self.output = torch.max(self.input, 2)
  end
  return self.output
end

function MultinomialAction:updateGradInput(input, gradOutput)
  local reward_loss = gradOutput
  
  self.gradInput:resizeAs(input):zero()
  self.gradInput:scatter(2, self.output, 1)
  self.gradInput:cdiv(self.input)
  
  local reward = reward_loss:expandAs(self.gradInput)
  self.gradInput:cmul(reward)
  
  return self.gradInput
end
