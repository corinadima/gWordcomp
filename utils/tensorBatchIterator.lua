
local tablex = require 'pl.tablex'

-- inspired by nextBatch function in
-- https://github.com/andresy/torch-demos/blob/master/logistic-regression/Trainer.lua

local tensorBatchIterator = {}
tensorBatchIterator.__index = tensorBatchIterator

function tensorBatchIterator.init(args)
	local self = {}
	setmetatable(self, tensorBatchIterator)

	self.data = args.data
	self.dataSize = args.data:size()
	self.inputShape = args.inputShape

	self.batchSize = args.batchSize
	self.withCUDA = args.cuda

	self.indices = torch.range(1, self.dataSize, 1) -- go through indices linearly
	self.index = 0

	if (args.randomize == true) then
		-- shuffe indices; this means that each pass sees the training examples in a different order 
		self.indices = torch.randperm(self.dataSize, 'torch.LongTensor')
	end

	return self
end

function tensorBatchIterator:processExample(example)
	return {{example[1][1]:clone(), example[1][2]:clone()}, example[2]:clone()}
end

-- batches are tensors
function tensorBatchIterator:nextBatch()
	if self.index >= self.dataSize then
		return nil
	end

	self.currentBatchSize = self.batchSize

	if (self.batchSize > self.dataSize - self.index) then
		self.currentBatchSize = self.dataSize - self.index
	end	

	local batch = {}

	local inputs1 = torch.Tensor(self.currentBatchSize, self.inputShape[2])
	local inputs2 = torch.Tensor(self.currentBatchSize, self.inputShape[2])
	local targets = torch.Tensor(self.currentBatchSize, self.data[1][2]:size()[1])

	local k = 1
	for i = self.index + 1, self.index + self.currentBatchSize do
		-- load new sample
		local example = self.data[self.indices[i]]
		processed = self:processExample(example)

		inputs1[k] = processed[1][1]
		inputs2[k] = processed[1][2]
		targets[k] = processed[2]

		k = k + 1
	end

	self.index = self.index + self.currentBatchSize


	if (self.withCUDA) then
		batch = {inputs={inputs1:cuda(), inputs2:cuda()}, targets=targets:cuda(), 
					currentBatchSize = self.currentBatchSize}
	else
		batch = {inputs={inputs1, inputs2}, targets=targets, currentBatchSize = self.currentBatchSize}
	end

	return batch
end

return tensorBatchIterator