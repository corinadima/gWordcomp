require 'paths' -- torch module for file path handling
dataLoader = require 'utils.dataLoader'

local compose_utils = {}

function compose_utils:loadDatasets(datasetDir, minNum)
	print('==> loading datasets...')
	
	local trainSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "train.txt"), " ")
	local devSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "dev.txt"), " ")
	local testSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "test.txt"), " ")
	local fullSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "full.txt"), " ")
	print('==> dataset loaded, train size:', trainSet:size(),
	  ' dev size', devSet:size(), ' test size', testSet:size(), ' full size', fullSet:size())

	return trainSet, devSet, testSet, fullSet
end	

function compose_utils:loadCMHDense(datasetDir, embeddingsId, size)
	print('==> loading dense embeddings of size ' .. size .. '...')
	local cmhEmbeddingsPath = paths.concat('data', datasetDir, 'embeddings', embeddingsId, embeddingsId .. '.' .. size .. 'd_cmh.dm')
	local cmhDictionary, cmhEmbeddings = dataLoader.loadDenseMatrix(cmhEmbeddingsPath)
	print('==> embeddings loaded, size:', cmhEmbeddings:size())

	return cmhDictionary, cmhEmbeddings
end

-- for Matrix, FullAdd
function compose_utils:createCMH2TensorDataset(tensorData, cmhEmbeddings)
	local dataset = {}

	local sz = cmhEmbeddings:size()[2]
		
	function dataset:size() return tensorData:size()[1] end
	function dataset:findEntry(compoundIndex)
		for i = 1, tensorData:size()[1] do
			if (tensorData[i][3] == compoundIndex) then
				return i
			end
		end		
		return nil		
	end
	for i = 1, tensorData:size()[1] do
		local input = torch.zeros(2, sz)
		input[1] = cmhEmbeddings[tensorData[i][1]]:clone()
		input[2] = cmhEmbeddings[tensorData[i][2]]:clone()


		local outputIndex = tensorData[i][3]
		local output =  cmhEmbeddings[outputIndex]:clone();
		dataset[i] = {input, output}
	end
	return dataset
end

return compose_utils
