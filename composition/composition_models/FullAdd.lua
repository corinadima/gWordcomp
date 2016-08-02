local compose_utils = require 'utils.compose_utils'
local batchIterator = require 'utils.tensorBatchIterator'

--
-- FullAdd composition model
--
-- 

local FullAdd, parent = torch.class('torch.FullAdd', 'torch.CompositionModel')

function FullAdd:__init(inputs, outputs)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
end

function FullAdd:architecture(config)
	print("# FullAdd; vector a and b are concatenated and composed through the global FullAdd W (size 2nxn);")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local c1 = nn.Identity()();
	local c2 = nn.Identity()();

	local W1 = nn.Linear(self.inputs/2, self.outputs)({c1})
	local W2 = nn.Linear(self.inputs/2, self.outputs)({c2})
	
	local added = nn.CAddTable(2)({W1, W2})
	
	self.mlp = nn.gModule({c1, c2}, {added})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function FullAdd:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH2TensorDataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH2TensorDataset(fullSet, cmhEmbeddings)
end

function FullAdd:train()
	print("Hyperparameter config")
	print(self.config)

	local config = self.config
	if (config.criterion == 'mse') then
		self.criterion = nn.MSECriterion()
		self.criterion.sizeAverage = true
	else
		error("Unknown criterion")
	end

	if (self.config.gpuid >= 0) then
		self.mlp:cuda()
		self.criterion:cuda()
	end

	local x, dl_dx = self.mlp:getParameters()
	self.bestModel = nil
	self.bestError = 2^20
	print("# FullAdd: training")

	function doTrain(module, reg, criterion, config, trainDataset, currentEpoch)
		module:training()

		-- Create a batched iterator
		local trainIter = batchIterator.init({
			data = trainDataset,
			randomize = true,
			cuda = true,
			batchSize = config.batchSize,
			inputShape = {2, self.inputs/2}
		})

		local nextBatch = trainIter:nextBatch()

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
				-- print("inputs:size()")
				-- print(inputs:size())
				-- print("outputs:size()")
				-- print(outputs:size())
				local loss_x = criterion:forward(outputs, targets)

				-- backward pass
				module:backward(inputs, criterion:backward(outputs, targets))

				-- return loss(x) amd dloss/dx
				return loss_x, dl_dx
			end

			-- optimize the current mini-batch
			if config.optimizer == 'adagrad' then
				_, fs = optim.adagrad(feval, x, config.adagrad_config, config.adagrad_config)
				trainError = trainError + fs[1] * currentBatchSize
				-- print(config.adagrad_config)
			else
				error('unknown optimization method')
			end

			nextBatch = trainIter:nextBatch()
		end

		-- report average error per epoch
		trainError = trainError/trainDataset:size()

		return trainError
	end

	function doTest(module, criterion, config, testDataset)
		module:evaluate()

		local testError = 0

		-- one pass 

		-- Create a batched iterator
		local testIter = batchIterator.init({
			data = testDataset,
			randomize = false,
			cuda = true,
			batchSize = config.batchSize,
			inputShape = {2, self.inputs/2}
		})

		local nextBatch = testIter:nextBatch()
		while (nextBatch ~= nil) do

			inputs = nextBatch.inputs
			targets = nextBatch.targets
			currentBatchSize = nextBatch.currentBatchSize

			-- test samples
			local predictions = module:forward(inputs)

			-- compute error
			err = criterion:forward(predictions, targets)
			testError = testError + err * currentBatchSize

			nextBatch = testIter:nextBatch()
		end

		-- average error over the dataset
		testError = testError/testDataset:size()

		return testError
	end


	self.epoch = 1
	local logger = optim.Logger(self.config.saveName, false) -- no timestamp
	logger.showPlot = false; -- if not run on a remote server, set this to true to show the learning curves in real time
	logger.plotRawCmd = 'set xlabel "Epochs"\nset ylabel "MSE"'
	logger.name = "Wmask"

	while true do
		itrainErr = doTrain(self.mlp, self.reg, self.criterion, self.config, self.trainDataset, self.epoch)
		trainErr = doTest(self.mlp, self.criterion, self.config, self.trainDataset)
		devErr = doTest(self.mlp, self.criterion, self.config, self.devDataset)
		self.epoch = self.epoch + 1
		print('Train error:\t', string.format("%.10f", trainErr))
		print('Dev error:\t', string.format("%.10f", devErr))
		print('Best error:\t', string.format("%.10f", self.bestError))

		-- log the errors for plotting
		logger:add{['training error'] = trainErr, ['dev error'] = devErr}
		logger:style{['training error'] = '-', ['dev error'] = '-'}
		logger:plot()

		-- early stopping when the error on the test set ceases to decrease
		if (self.config.earlyStopping == true) then
			if (devErr < self.bestError) then
				self.bestError = devErr
				self.bestModel = self.mlp:clone()
				self.extraIndex = 1
			else
				if (self.extraIndex < self.config.extraEpochs) then
					self.extraIndex = self.extraIndex + 1
				else
					print("# Composer: stopping - you have reached the maximum number of epochs after the best model")
					print("# Composer: best error: " .. self.bestError)
					self.mlp = self.bestModel:clone()
					break					 
				end
			end
		end
	end
 end

 function FullAdd:predict(onDev, onTest, onFull, cmhDictionary, devSet, testSet, fullSet)
 	function predict(module, config, dataset)
 		module:evaluate()
 		local predictions = torch.Tensor(dataset:size(), dataset[1][2]:size()[1])

 		-- one pass through the data
		local dataIter = batchIterator.init({
			data = dataset,
			randomize = false,
			cuda = true,
			batchSize = config.batchSize,
			inputShape = {2, self.inputs/2}
		})

		local t = 1
		local nextBatch = dataIter:nextBatch()
		while (nextBatch ~= nil) do

			inputs = nextBatch.inputs
			targets = nextBatch.targets
			currentBatchSize = nextBatch.currentBatchSize

			local preds = module:forward(inputs):float()
			predictions[{{t, t + currentBatchSize - 1}, {}}] = preds
			t = t + currentBatchSize

			nextBatch = dataIter:nextBatch()
		end
		return predictions
 	end

	function savePredictions(cmhDictionary, predictions, saveName, indexSet, separator)
		local field_delim = separator or ' '
		local outputFileName = saveName .. '.pred'
		local f = io.open(outputFileName, "w")
		-- print(indexSet:size())
		-- print(indexSet[1])
		-- print("predictions", predictions:size())

		for i = 1, predictions:size()[1] do
			local cidx = indexSet[i][3]
			local word = cmhDictionary[cidx]

			f:write(word .. field_delim)
			for j = 1,predictions:size()[2] do
				f:write(string.format("%.6f", predictions[i][j]))
				if (j < predictions:size()[2]) then
				  f:write(field_delim)
				end
			end
			f:write("\n")
		end
		f:close()
	end

 	if (onDev == true) then
		print(" # Creating dev set predicitons... ")
		local devPredictions = predict(self.mlp, self.config, self.devDataset)
		savePredictions(cmhDictionary, devPredictions, self.config.saveName .. '_dev', devSet, ' ')
	end

	if (onTest == true) then
		print(" # Creating test set predicitons... ")
		local testPredictions = predict(self.mlp, self.config, self.testDataset)
		savePredictions(cmhDictionary, testPredictions, self.config.saveName .. '_test', testSet, ' ')
	end

	if (onFull == true) then
		print(" # Creating full predicitons... ")
		local fullPredictions = predict(self.mlp, self.config, self.fullDataset)
		savePredictions(cmhDictionary, fullPredictions, self.config.saveName .. '_full', fullSet, ' ')
	end	
end