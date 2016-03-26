--
--  glimpse.lua
--  hard-attention
--
--  Created by Andrey Kolishchak on 09/27/15.
--
require 'torch'

function init_glimpse_data(x, width, step)
  local r_steps = 1 + (x:size(3)-width)/step
  local c_steps = 1 + (x:size(4)-width)/step
  local data = x.new(x:size(1), x:size(2), r_steps, c_steps, width*width)
  for gr=1,r_steps do
    for gc=1,c_steps do
      local gr_start = (gr-1)*step + 1
      local gc_start = (gc-1)*step + 1
      data[{{},{},gr,gc}] = x:sub(1, x:size(1), 1, x:size(2), gr_start, gr_start+width-1, gc_start, gc_start+width-1)
    end
  end
  
  return data
end

function get_glimpse(data, location)
  local result = data.new(data:size(1), data:size(2), data:size(5))
  for i=1,data:size(1) do
    for channel=1,data:size(2) do
      result[i][channel] = data[i][channel][location[i][1]][location[i][2]]
    end
  end
  
  return result:view(data:size(1),-1)
end

local function test()
  local x = torch.Tensor({
                        {11,12,13,14,15,16},
                        {21,22,23,24,25,26},
                        {31,32,33,34,35,36},
                        {41,42,43,44,45,46},
                        {51,52,53,54,55,56},
                        {61,62,63,64,65,66},
                      })

  local y = torch.Tensor(3, 1, x:size(1), x:size(2))
  y[1][1]:copy(x)
  y[2][1]:copy(x):mul(10)
  y[3][1]:copy(x):mul(100)

  local data = init_glimpse_data(y, 2, 2)
  print(data:view(data:size(1), data:size(3)*data:size(4), -1))
  print(get_glimpse(data, {{1,2},{3,1},{2,3}}):squeeze())
end

--test()
