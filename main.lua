--
--  main.lua
--  hard-attention
--
--  Created by Andrey Kolishchak on 09/27/15.
--
require 'nn'
require 'optim'
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'image'

require 'data.dataset'
require 'glimpse'
require 'MultinomialAction'

local model_utils = require 'util.model_utils'
local LSTM = require 'LSTM'
local action_policy = require 'action_policy'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Hard Attention')
cmd:text()
cmd:text('Options')
cmd:option('-glimpse_width', 4, 'width of glimpse')
cmd:option('-glimpse_step', 2, 'width of glimpse')
cmd:option('-rnn_size', 512, 'size of RNN internal state')
cmd:option('-seq_length', 20, 'number of timesteps to unroll for')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-dropout', 0.9, 'dropout')
cmd:option('-learning_rate', 1e-3, 'learning rate')
cmd:option('-batch_size', 100, 'number of sequences to train on in parallel')
cmd:option('-max_epoch', 10, 'number of full passes through the training data')
cmd:option('-gpu',2,'0 - cpu, 1 - cunn, 2 - cudnn')
cmd:option('-output_path', 'images', 'path for output images')
cmd:option('-pass_type', 'train', 'type of pass: train, test')

opt = cmd:parse(arg)
opt.glimpse_size = opt.glimpse_width * opt.glimpse_width

if opt.gpu > 0 then
  require 'cunn'
  if opt.gpu == 2 then
    require 'cudnn'
  end
end

--
-- load data
--
print("loading data...")
local dataset = load_mnist(opt)
dataset.train_x_glimpse = init_glimpse_data(dataset.train_x, opt.glimpse_width, opt.glimpse_step)
dataset.test_x_glimpse = init_glimpse_data(dataset.test_x, opt.glimpse_width, opt.glimpse_step)

opt.image_width = dataset.train_x:size(3)
opt.max_step = dataset.train_x_glimpse:size(3)
opt.location_image_size = (opt.image_width/opt.glimpse_width)^2
opt.channel_num = dataset.train_x:size(2)

--
-- build model
--
print("building model...")
local rnn_model = LSTM.lstm(opt.glimpse_size, opt.rnn_size, opt.num_layers, opt.dropout)

local action_model = nn.Sequential()
action_model:add(nn.SelectTable(-1)) -- take rnn's top h state
action_model:add(nn.Linear(opt.rnn_size, opt.rnn_size))
action_model:add(nn.ELU())
if opt.dropout > 0 then action_model:add(nn.Dropout(opt.dropout)) end
action_model:add(nn.Linear(opt.rnn_size, 8))
action_model:add(nn.ELU())
action_model:add(nn.SoftMax())
action_model:add(nn.MultinomialAction())

local glimpse_model = nn.Sequential()
glimpse_model:add(rnn_model)
glimpse_model:add(nn.ConcatTable()
                    :add(nn.Identity()) -- rnn states
                    :add(action_model) -- actions
                )

local class_model = nn.Sequential()
class_model:add(nn.SelectTable(-1)) -- take rnn's top h state
class_model:add(nn.Linear(opt.rnn_size, 10))
class_model:add(nn.LogSoftMax())

function get_location_dim_model()
  local model = nn.Sequential()
  model:add(nn.Linear(opt.location_image_size*opt.channel_num, opt.max_step))
  model:add(nn.ELU())
  model:add(nn.SoftMax())
  model:add(nn.MultinomialAction())
  return model
end

local initial_location_model = nn.Sequential()
initial_location_model:add(nn.SpatialMaxPooling(opt.glimpse_width,opt.glimpse_width,opt.glimpse_width,opt.glimpse_width)) -- downsize the original image
initial_location_model:add(nn.Reshape(opt.location_image_size*opt.channel_num))
initial_location_model:add(nn.Dropout(opt.dropout))
initial_location_model:add(nn.ConcatTable()
                              :add(get_location_dim_model())
                              :add(get_location_dim_model())
                          )
initial_location_model:add(nn.JoinTable(2))


local criterion = nn.ClassNLLCriterion()

if opt.gpu > 0 then
  glimpse_model:cuda()
  class_model:cuda()
  criterion:cuda()
  initial_location_model:cuda()
  if opt.gpu == 2 then
    cudnn.convert(glimpse_model, cudnn)
    cudnn.convert(criterion, cudnn)
    cudnn.benchmark = true
  end
end

-- the initial state of rnn
local glimpse_h_init = {}
local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
for l=1,opt.num_layers do
    if opt.gpu > 0 then
      h_init = h_init:cuda()
    end
    table.insert(glimpse_h_init, h_init:clone())
    table.insert(glimpse_h_init, h_init:clone())
end

-- 1:up 2:up-right 3:right 4:down-right 5:down 6:down-left 7:left 8:up-left
local action_offset = torch.LongTensor({
        {-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}
      }):repeatTensor(opt.batch_size, 1, 1)

params, grad_params = model_utils.combine_all_parameters(glimpse_model, class_model, initial_location_model)
--params:uniform(-0.08, 0.08)

glimpse_model_clones = model_utils.clone_many_times(glimpse_model, opt.seq_length)

action_policy.init(opt)

--
-- optimize
--
local iterations = opt.max_epoch*dataset.train_x_glimpse:size(1)/opt.batch_size
local batch_start = 1

function feval(x)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()

  -- load batch
  local input = dataset.train_x_glimpse[{{batch_start, batch_start+opt.batch_size-1},{}}]
  local target = dataset.train_y[{{batch_start, batch_start+opt.batch_size-1}}]
  local input_image = dataset.train_x[{{batch_start, batch_start+opt.batch_size-1},{}}]
  
  -- forward pass
  
  -- initial location
  local location = initial_location_model:forward(input_image):long()
  
  local glimpse = {}
  local glimpse_h = { [0] = glimpse_h_init }
  local daction = {}
  for t=1,opt.seq_length do
    -- take glimpse
    glimpse[t] = get_glimpse(input, location)
    -- rnn step
    glimpse_model_clones[t]:training()
    glimpse_h[t], action = unpack(glimpse_model_clones[t]:forward{glimpse[t], unpack(glimpse_h[t-1])})
    -- take action
    local index = torch.LongTensor(opt.batch_size,1,2)
    action = action:long()--:clamp(1,8)
    index[{{},{},{1}}] = action
    index[{{},{},{2}}] = action
    location:add(action_offset:gather(2, index)):clamp(1, opt.max_step)
  end
  
  local glimpse_output = glimpse_h[#glimpse_h]
  
  local class = class_model:forward(glimpse_output)
  local loss = criterion:forward(class, target)
  local reward_loss = action_policy.get_reward_loss(class, target)
  
  -- backward pass
  local dloss_dcriterion = criterion:backward(class, target)
  local dloss_dclass = class_model:backward(glimpse_output, dloss_dcriterion)
    
  dclass_dglimpse_h = { [opt.seq_length] = dloss_dclass }
  
  for t=opt.seq_length,1,-1 do
    dclass_dglimpse_h[t-1] = glimpse_model_clones[t]:backward({glimpse[t], unpack(glimpse_h[t-1])}, {dclass_dglimpse_h[t], reward_loss})
    table.remove(dclass_dglimpse_h[t-1], 1) -- remove x gradient
  end
  
  -- transfer final state to initial state
  glimpse_h_init = glimpse_h[#glimpse_h]
  
  initial_location_model:backward(input_image, reward_loss:expand(reward_loss:size(1),2))
    
  return loss+reward_loss:mean(), grad_params
  --return loss, grad_params
end

--
-- training
--
class_model:training()
initial_location_model:training()
local optim_state = {learningRate = opt.learning_rate}
print("trainig...")

for it = 1,iterations do
  
    local _, loss = optim.adam(feval, params, optim_state)

    if it % 100 == 0 then
      print(string.format("batch = %d, loss = %.12f", it, loss[1]))
    end
  
    batch_start = batch_start + opt.batch_size
    if batch_start > dataset.train_x_glimpse:size(1) then
      batch_start = 1
    end 
    
end

print("evaluating...")
class_model:evaluate()
initial_location_model:evaluate()

paths.mkdir(opt.output_path)

function get_loss(x, x_image, y, log_fails, draw_actions)
  local match = 0.0
  local draw_actions_images = 0
  for i=1,x:size(1),opt.batch_size do  
    local input = x[{{i, i+opt.batch_size-1},{}}]
    local target = y[{{i, i+opt.batch_size-1}}]
    local input_image = x_image[{{i, i+opt.batch_size-1},{}}]

      -- initial location
    local location = initial_location_model:forward(input_image):long()
  
    local glimpse = {}
    local glimpse_h = { [0] = glimpse_h_init }
    local loc = {}
    local act = {}
    for t=1,opt.seq_length do
      
      loc[t] = location:clone()
      -- take glimpse
      glimpse[t] = get_glimpse(input, location)
      -- rnn step
      glimpse_model_clones[t]:evaluate()
      glimpse_h[t], action = unpack(glimpse_model_clones[t]:forward{glimpse[t], unpack(glimpse_h[t-1])})
      -- take action
      local index = torch.LongTensor(opt.batch_size,1,2)
      action = action:long()
      index[{{},{},{1}}] = action
      index[{{},{},{2}}] = action
      location:add(action_offset:gather(2, index)):clamp(1, opt.max_step)
      
      act[t] = action
    end
  
    local glimpse_output = glimpse_h[#glimpse_h]
  
    local class = class_model:forward(glimpse_output)

    prob, idx = torch.max(class, 2)
    
    match = match + torch.mean(idx:eq(target):float())/(x:size(1)/opt.batch_size)
    
    if log_fails == true then
      local matches = idx:eq(target)
      for j=1,matches:size(1) do
        if matches[j][1] == 0 then
          local k = i-1+j
          image.save(opt.output_path..'/fail_'..tostring(k)..'-'..tostring(y[k])..'-'..tostring(idx[j][1])..'.jpg',x[{{k},{}}]:view(28,28))
        end 
      end
    end
    
    if draw_actions == true and draw_actions_images < 300 then
      local matches = idx:eq(target)
      local step = opt.glimpse_step
      local width = opt.glimpse_width
      for j=1,matches:size(1) do
        if matches[j][1] == 1 and draw_actions_images < 300 then
          local img_index = i-1+j
          local img = x_image[{{img_index},{}}]:view(28,28)
          for k,v in pairs(loc) do
            local gr_start = (v[j][1]-1)*step + 1
            local gc_start = (v[j][2]-1)*step + 1
            img[{{gr_start, gr_start+width-1},{gc_start, gc_start+width-1}}]:fill(k==1 and 0.5 or 0.3)
          end
          image.save(opt.output_path..'/actions_'..tostring(img_index)..'-'..tostring(idx[j][1])..'.jpg',img)
          draw_actions_images = draw_actions_images + 1
        end 
      end
    end
    
  end

  return match
end

print(string.format("training = %.2f%%, testing = %.2f%%",
    get_loss(dataset.train_x_glimpse, dataset.train_x, dataset.train_y, false, true)*100.0,
    get_loss(dataset.test_x_glimpse, dataset.test_x, dataset.test_y, false, false)*100.0))
