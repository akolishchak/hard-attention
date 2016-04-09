--
--  demo.lua
--  hard-attention
--
--  Created by Andrey Kolishchak on 4/3/16.
--

function print_image(img)
  local str = "    "
  for i=1,img:size(2) do
    local v = math.fmod(i, 10)
    str = str .. tostring(v) .. " "
  end
  print(str);
  
  for i = 1,img:size(1) do
    str = string.format("%2d: ", i)
    for j = 1,img:size(2) do
      local v = img[i][j] < 0 and "0" or "1"
      str = str .. v .. " "
    end
    print(str)
  end
end

function get_demo(data, opt)

  local img = data[1][1]

  local location = {3, 11}
  local offsets = {{0, 0}, {0,-1}, {1,-1}, {0,-1}, {0,-1}, {0,-1}, {0,-1},
                   {1,1},  {1,1},  {1,0},  {0,1},  {1,1},  {1,0},  {1,-1},
                   {0,-1}, {1,-1}, {1,-1}, {0,-1}, {0,-1}, {0, 1}}
  
  local demo = img.new(opt.seq_length, img:size(3))

  for i=1,#offsets do
    location[1] = location[1] + offsets[i][1]
    location[2] = location[2] + offsets[i][2]
    demo[i] = img[location[1]][location[2]]
    --print_image(demo[i]:view(opt.glimpse_width, opt.glimpse_width))
  end

  return demo:view(opt.seq_length, 1, -1):expand(opt.seq_length, opt.batch_size, img:size(3))
end
