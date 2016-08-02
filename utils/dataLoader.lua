require 'torch'

local stringx = require('pl.stringx')

-- data loading utilities

local dataLoader = {}

function dataLoader.loadDenseMatrix(fileName, separator)
  local dict = {}
  local dataset = {}
  local dt_index = 0

  local field_delim = separator or ' '

  function dataset:size()
    return dt_index
  end

  local field_delim = separator or ' '

  io.input(fileName)
  local lines = io.lines()
  local loaded_tensor = nil

  local load_limit = limit or -1 
  for line in io.lines() do
    local sp = stringx.split(stringx.strip(line), field_delim)
    local tensor_size = #sp-1
    local loaded_tensor = torch.DoubleTensor(tensor_size):zero()
    table.insert(dict, stringx.strip(sp[1]))
    local index = 2
    while index <= #sp do
      loaded_tensor[index-1] = sp[index]
      index = index + 1
    end

    dt_index = dt_index + 1;
    table.insert(dataset, loaded_tensor)
  end
  io.close()

  local datasetTensor = torch.Tensor(dataset:size(), dataset[1]:size()[1])
  for i = 1, dataset:size() do
    datasetTensor[i] = dataset[i]
  end

  return dict, datasetTensor
end

-- function dataLoader.loadDictionary( fileName )
--   local dict = {}
--   io.input(fileName)
--   local idx = 1
--   for line in io.lines() do
--     table.insert(dict, stringx.strip(line))
--   end
--   io.close()

--   function  dict:getIndex(string)
--     local index = -1
--     local i = 1
--     while i <= #dict do
--       if (dict[i] == string) then
--         index = i
--         break
--       end
--       i = i + 1
--     end
--     return index
--   end

--   return dict
-- end

function dataLoader.loadSimpleDataset(fileName, separator)
  local dataset = {}
  local dt_index = 0

  function dataset:size()
    return dt_index
  end

  local field_delim = separator or ' '

  io.input(fileName)
  local lines = io.lines()
  local loaded_tensor = nil

  local load_limit = limit or -1 
  for line in io.lines() do
  	local sp = stringx.split(stringx.strip(line), field_delim)
  	local tensor_size = #sp
  	local loaded_tensor = torch.DoubleTensor(tensor_size):zero()
  	local index = 1
  	while index <= #sp do
  		loaded_tensor[index] = sp[index]
  		index = index + 1
  	end

  	dt_index = dt_index + 1;
    table.insert(dataset, loaded_tensor)
  end
  io.close()

  local datasetTensor = torch.Tensor(dataset:size(), dataset[1]:size()[1])
  for i = 1, dataset:size() do
    datasetTensor[i] = dataset[i]
  end

  return datasetTensor
end

return dataLoader
