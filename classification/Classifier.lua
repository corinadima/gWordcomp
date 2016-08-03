require 'torch'
require 'nn'
require 'optim'

local Classifier = torch.class('nn.Classifier')

local tablex = require('pl.tablex')

------------------------------------------------------
------------------------------------------------------
local BatchIterator = {}
BatchIterator.__index = BatchIterator

function BatchIterator:init(args)
	local self = {}
	setmetatable(self, BatchIterator)

	self.data = {}
	for i = 1, tablex.size(args.data) do
		if (type(args.data[i]) ~= 'function') and (args.data[i] ~= nil) then
			table.insert(self.data, args.data[i])
		end
	end

	self.dataSize = tablex.size(self.data)

	self.inputShape = args.inputShape

	self.batchSize = args.batchSize
	self.withCUDA = args.cuda

	self.indices = torch.range(1, self.dataSize, 1) -- go through indices linearly
	self.index = 0

	self.dualInput = args.dualInput
	self.quadrupleInput = args.quadrupleInput
	self.pretrainedModel = args.pretrainedModel
	self.pretrainedModel2 = args.pretrainedModel2

	if (args.randomize == true) then
		-- shuffe indices; this means that each pass sees the training examples in a different order 
		self.indices = torch.randperm(self.dataSize, 'torch.LongTensor')
	end

	return self	
end

function BatchIterator:processExample(example)
	-- input is 2x300 dim, output is number (class index)
	return {example[1]:clone(), example[2]}
end

function BatchIterator:nextBatch()
	if self.index >= self.dataSize then
		return nil
	end

	self.currentBatchSize = self.batchSize

	if (self.batchSize > self.dataSize - self.index) then
		self.currentBatchSize = self.dataSize - self.index
	end	

	local batch = {}

	local inputs = torch.Tensor(self.currentBatchSize, self.inputShape[1], self.inputShape[2])
	local targets = torch.Tensor(self.currentBatchSize)

	local k = 1
	for i = self.index + 1, self.index + self.currentBatchSize do
		-- load new sample
		local example = self.data[self.indices[i]]
		processed = self:processExample(example)

		inputs[k] = processed[1]
		targets[k] = processed[2]

		k = k + 1
	end

	self.index = self.index + self.currentBatchSize


	if (self.withCUDA) then
		batch = {inputs=inputs:cuda(), targets=targets:cuda(), currentBatchSize = self.currentBatchSize}
	else
		batch = {inputs=inputs, targets=targets, currentBatchSize = self.currentBatchSize}
	end

	return batch
end

function BatchIterator:nextBatchTable()
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
	local targets = torch.Tensor(self.currentBatchSize)

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


	local f_inputs = nil
	local out_inputs = nil
	local f_targets = nil

	if (self.withCUDA) then
		f_inputs = {inputs1:cuda(), inputs2:cuda()}
		out_inputs = {inputs1:cuda(), inputs2:cuda()}
		f_targets=targets:cuda()
	else 
		f_inputs = {inputs1, inputs2}
		out_inputs = {inputs1, inputs2}
		f_targets = targets
	end

	if (self.pretrainedModel ~= nil) then
		-- we have a pretrained model; pass the inputs though it first and return the result as the new input
		local ptOutputs = self.pretrainedModel:forward(f_inputs)		
		if (self.dualInput == true) then
			table.insert(out_inputs, ptOutputs)
		else
			out_inputs = ptOutputs
		end
	end

	if (self.pretrainedModel2 ~= nil) then
		-- we have a pretrained model; pass the inputs though it first and return the result as the new input
		local pt2Outputs = self.pretrainedModel2:forward(f_inputs)		
		if (self.quadrupleInput == true) then
			table.insert(out_inputs, pt2Outputs)
		else
			out_inputs = pt2Outputs
		end
	end

	batch = {inputs=out_inputs, targets=f_targets, currentBatchSize = self.currentBatchSize}

	return batch
end

------------------------------------------------------
------------------------------------------------------

function Classifier:init(inputs, classes, pretrainedModel, pretrainedModel2)
	self.inputs = inputs
	self.classes = classes
	self.pretrainedModel = pretrainedModel
	self.pretrainedModel2 = pretrainedModel2
	self.dualInput = false
	self.quadrupleInput = false

	return self
end

function Classifier:architecture(mlp, config)
	self.mlp = mlp
	self.config = config
end

function Classifier:data(trainSet, devSet)
	self.trainDataset = trainSet
	self.devDataset = devSet
end

function Classifier:train()
	print("Hyperparameter config")
	print(self.config)

	self.criterion = nn.ClassNLLCriterion()
	
	if (self.config.gpuid >= 0) then
		self.mlp:cuda()
		self.criterion:cuda()
	end

	local x, dl_dx = self.mlp:getParameters()
	print("# Classifier: training")

	local optim_config = nil
	if self.config.optimizer == 'adagrad' then
		optim_config = {
			learningRate = self.config.adagrad_config.learningRate,
			learningRateDecay = self.config.adagrad_config.learningRateDecay,
			weightDecay = self.config.adagrad_config.weightDecay
		}
	elseif self.config.optimizer == 'rmsprop' then
		optim_config = {
			learningRate = self.config.rmsprop_config.learningRate,
			alpha = self.config.rmsprop_config.alpha,
			epsilon = self.config.rmsprop_config.epsilon,
			wd = self.config.rmsprop_config.wd

		}
	elseif self.config.optimizer == 'asgd' then
		optim_config = self.config.asgd_config
	else
		error('Unknown optimizer')
	end

	local optim_state = optim_config


	function eval(confusion_m)
		local classes = confusion_m.classes
		local confusion = confusion_m.mat:double()
		-- parse matrix / normalize / count scores
		local diag = torch.FloatTensor(#classes)
		local freqs = torch.FloatTensor(#classes)
		local unconf = confusion
		local confusion = confusion:clone()
		local corrects = 0
		local total = 0
		for target = 1,#classes do
		  freqs[target] = confusion[target]:sum()
		  corrects = corrects + confusion[target][target]
		  total = total + freqs[target]
		  confusion[target]:div( math.max(confusion[target]:sum(),1) )
		  diag[target] = confusion[target][target]
		end

		-- accuracies
		local accuracy = corrects / total * 100
		local perclass = 0
		local total = 0
		for target = 1,#classes do
		  if confusion[target]:sum() > 0 then
		     perclass = perclass + diag[target]
		     total = total + 1
		  end
		end
		perclass = perclass / total * 100
		freqs:div(unconf:sum())	

		local classFrrs, classFars, returnFrrs, returnFars = confusion_m:farFrr()

		return accuracy
	end

	function doTrain(module, criterion, config, trainDataset, currentEpoch, confusion)
		module:training()

		-- Create a batched iterator
		local trainIter = BatchIterator:init({
			data = trainDataset,
			randomize = true,
			cuda = true,
			batchSize = config.batchSize,
			inputShape = {2, self.inputs},
			pretrainedModel = self.pretrainedModel,
			pretrainedModel2 = self.pretrainedModel2,
			dualInput = self.dualInput,
			quadrupleInput = self.quadrupleInput,
		})

		-- local nextBatch = trainIter:nextBatch()
		local nextBatch = trainIter:nextBatchTable()

		print("# Epoch " .. currentEpoch)
		local trainError = 0

		-- one epoch
		while (nextBatch ~= nil) do

			inputs = nextBatch.inputs
			targets = nextBatch.targets

			currentBatchSize = nextBatch.currentBatchSize

			-- create eval clojure
			local feval = function(x_new)
				collectgarbage()

				-- get the new parameters
				if x ~= x_new then
					x:copy(x_new)
				end

				-- reset gradients
				dl_dx:zero()

				-- evaluate on complete mini-batch

				-- forward pass
				local outputs = module:forward(inputs)
				local loss_x = criterion:forward(outputs, targets)

				-- backward pass
				module:backward(inputs, criterion:backward(outputs, targets))

				confusion:batchAdd(outputs, targets)

				if config.grad_clip > 0 then
    				dl_dx:clamp(-config.grad_clip, config.grad_clip)
  				end

				-- return loss(x) amd dloss/dx
				return loss_x, dl_dx
			end

			-- optimize the current mini-batch
			if config.optimizer == 'adagrad' then
				_, fs = optim.adagrad(feval, x, optim_config, optim_state)
				-- if currentEpoch == 1 then
				-- 	print(config.adagrad_config)
				-- end
			elseif config.optimizer == 'adam' then
				_, fs = optim.adam(feval, x, optim_config, optim_state)
			elseif config.optimizer == 'rmsprop' then
				_, fs = optim.adagrad(feval, x, optim_config, optim_state)
				if currentEpoch == 1 then
					print(config.rmsprop_config)
				end
			elseif config.optimizer == 'asgd' then
				_, fs = optim.asgd(feval, x, config.asgd_config, config.asgd_config)
			else
				error('unknown optimization method')
			end

			trainError = trainError + fs[1] * currentBatchSize
			nextBatch = trainIter:nextBatchTable()
		end

		trainAccuracy = eval(confusion)
		print("Train accuracy " .. trainAccuracy)
		confusion:zero()

		-- report average error per epoch
		trainError = trainError/tablex.size(trainDataset)

		return trainAccuracy, trainError
	end

	function doTest(module, criterion, config, testDataset, confusion)
		module:evaluate()

		local testError = 0

		-- one pass 

		-- Create a batched iterator
		local testIter = BatchIterator:init({
			data = testDataset,
			randomize = false,
			cuda = true,
			batchSize = config.batchSize,
			inputShape = {2, self.inputs},
			pretrainedModel = self.pretrainedModel,
			pretrainedModel2 = self.pretrainedModel2,
			dualInput = self.dualInput,
			quadrupleInput = self.quadrupleInput,
		})

		-- local nextBatch = testIter:nextBatch()
		local nextBatch = testIter:nextBatchTable()
		while (nextBatch ~= nil) do

			inputs = nextBatch.inputs
			targets = nextBatch.targets
			currentBatchSize = nextBatch.currentBatchSize

			-- test samples
			local predictions = module:forward(inputs)

			-- compute error
			err = criterion:forward(predictions, targets)
			testError = testError + err * currentBatchSize
			confusion:batchAdd(predictions, targets)

			-- nextBatch = testIter:nextBatch()
			nextBatch = testIter:nextBatchTable()
		end

		testAccuracy = eval(confusion)
		print("Test accuracy " .. testAccuracy)
		confusion:zero()

		-- average error over the dataset
		testError = testError/tablex.size(testDataset)

		return testAccuracy, testError
	end

	function predict(module, testDataset)
		-- Create a batched iterator
		local dataIter = BatchIterator:init({
			data = testDataset,
			randomize = false,
			cuda = true,
			batchSize = self.config.batchSize,
			inputShape = {2, self.inputs},
			pretrainedModel = self.pretrainedModel,
			pretrainedModel2 = self.pretrainedModel2,
			dualInput = self.dualInput,
			quadrupleInput = self.quadrupleInput,
		})

		print(dataIter.index)

		local nextBatch = dataIter:nextBatchTable()
		local predictions = torch.Tensor(dataIter.dataSize,1)

		local t = 1
		while (nextBatch ~= nil) do
			inputs = nextBatch.inputs
			targets = nextBatch.targets
			currentBatchSize = nextBatch.currentBatchSize

			-- test samples
			local preds = module:forward(inputs):float()
			local y, idxs = torch.max(preds, 2)
			predictions[{{t, t + currentBatchSize - 1}, {}}] = idxs

			t = t + currentBatchSize
			nextBatch = dataIter:nextBatchTable()
		end		

		return predictions
	end

	--------------------------------------
	-- and train!

	self.epoch = 1
	self.bestModel = nil
	self.bestError = 2^20
	self.bestAccuracy = -2^20

	-- This matrix records the current confusion across classes
	self.confusion = optim.ConfusionMatrix(self.classes)

	-- training loop with early stopping
	while true do
		itrainAcc, itrainErr = doTrain(self.mlp, self.criterion, self.config, self.trainDataset, self.epoch, self.confusion)
		devAcc, devErr = doTest(self.mlp, self.criterion, self.config, self.devDataset, self.confusion)
		self.epoch = self.epoch + 1

		-- early stopping when the error on the test set ceases to decrease
		if (self.config.earlyStopping == true) then
			if (devAcc > self.bestAccuracy) then
				self.bestAccuracy = devAcc
				self.bestError = devErr
				self.bestModel = self.mlp:clone()
				self.extraIndex = 1
			else
				if (self.extraIndex < self.config.extraEpochs) then
					self.extraIndex = self.extraIndex + 1
				else
					print("# Composer: stopping - you have reached the maximum number of epochs after the best model")
					print("# Composer: best accuracy: " .. self.bestAccuracy)
					self.mlp = self.bestModel:clone()

					local predictions = predict(self.mlp, self.devDataset)

					return self.bestAccuracy, self.mlp, predictions

				end
			end
		end
	end
end

return Classifier