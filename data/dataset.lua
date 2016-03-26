
function load_mnist(opt)
  
  local mnist = require 'mnist'
  local traindataset = mnist.traindataset()
  local testdataset = mnist.testdataset()

  -- fix labels
  traindataset.label:add(traindataset.label:eq(0):mul(10))
  testdataset.label:add(testdataset.label:eq(0):mul(10))

  local dataset = {}
  -- add channels and convert to float
  dataset.train_x = traindataset.data:view(traindataset.data:size(1), 1, traindataset.data:size(2),      traindataset.data:size(3)):double()
  dataset.train_y = traindataset.label:double()

  dataset.test_x = testdataset.data:view(testdataset.data:size(1), 1, testdataset.data:size(2), testdataset.data:size(3)):double()
  dataset.test_y = testdataset.label:double()

  if opt.gpu > 0 then
    dataset.train_x = dataset.train_x:cuda()
    dataset.train_y = dataset.train_y:cuda()
    dataset.test_x = dataset.test_x:cuda()
    dataset.test_y = dataset.test_y:cuda()
  end
  
  -- normalize
  local mean = dataset.train_x:mean()
  local std = dataset.train_x:std()
  dataset.train_x:add(-mean):div(std)
  dataset.test_x:add(-mean):div(std)
  
  return dataset
end
