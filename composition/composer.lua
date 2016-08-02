require 'optim'

-- class for training a composition model
-- code based on the Torch supervised learning tutorial (https://github.com/torch/tutorials/tree/master/2_supervised)

local Composer = torch.class('nn.Composer')

function Composer:__init(m, config)
	self.module = m
	
	if (config.criterion == 'abs') then
		self.criterion = nn.AbsCriterion()
	elseif (config.criterion == 'mse') then
		self.criterion = nn.MSECriterion()
	else
		error("Unknown criterion.")
	end

	self.config = config
	print(self.config)
end

function Composer:train(trainDataset, devDataset)
	local x, dl_dx = self.module:getParameters()	
	self.bestModel = nil
	self.bestError = 2^20
	print("# Composer: training")

	function doTrain(module, criterion, config, trainDataset)
		module:training()

		epoch = epoch or 1
		print("# Epoch ", epoch)
		local trainError = 0

		-- shuffe indices; this means that each pass sees the training examples in a different order 
		local shuffledIndices = torch.randperm(trainDataset:size(), 'torch.LongTensor')

		-- one epoch
		for t = 1, trainDataset:size(), config.batchSize do

			-- create mini-batch
			local inputs = {}
			local targets = {}
			for i = t, math.min(t + config.batchSize - 1, trainDataset:size()) do
				-- load new sample
				local example = trainDataset[shuffledIndices[i]]
				local input = example[1]
				local target = example[2]

				table.insert(inputs, input)
				table.insert(targets, target)
			end

			-- create clojure to evaluate function and its derivative on the mini-batch
			local feval = function(x_new)
				-- just in case:
				collectgarbage()			

				-- get new parameters 
				if x ~= x_new then
					x:copy(x_new)
				end

				-- reset gradients
				dl_dx:zero()

				-- evaluate on complete mini-batch
				-- forward pass
				local outputs = module:forward(inputs)
				local loss_x = criterion:forward(outputs, targets)

				--backward pass
				module:backward(inputs, criterion:backward(outputs, targets))

			    -- weight decay (l2 regularization)
			    loss_x = loss_x + 0.5 * config.weightDecay * x:norm() ^ 2
			    dl_dx:add(config.weightDecay, x)

			    -- return loss(x) and dloss/dx
			    return loss_x, dl_dx
			end	

			-- optimize the current mini-batch
			if config.optimizer == 'adagrad' then
				_, fs = optim.adagrad(feval, x, config.adagrad_config)
				trainError = trainError + fs[1]
			else 
				error('unknown optimization method')
			end
		end

	    -- report average error on epoch
	    train_error = train_error/trainDataset:size()

		-- next epoch
		epoch = epoch + 1

		return trainError
	end

	function doTest(module, criterion, config, dataset)
		module:evaluate()
		local testError = 0

		for t = 1, dataset:size(), config.batchSize do

			-- create mini-batch
			local inputs = {}
			local targets = {}
			for i = t, math.min(t + config.batchSize - 1, dataset:size()) do
				-- load new sample
				local example = dataset[i]
				local input = example[1]
				local target = example[2]

				table.insert(inputs, input)
				table.insert(targets, target)
			end

			-- test samples
			local preds = module:forward(inputs)

			-- compute error
			err = criterion:forward(preds, targets)
			testError = testError + err
		end

		-- average error over the dataset
		testError = testError/dataset:size()

		return testError
	end

	while true do
		trainErr = doTrain(self.module, self.criterion, self.config, trainDataset)
		testErr = doTest(self.module, self.criterion, self.config, devDataset)
		print('Train error:\t', string.format("%.6f", trainErr))
		print('Test error:\t', string.format("%.6f", testErr))
		print('Best error:\t', string.format("%.6f", self.bestError))

		-- early stopping when the error on the test set ceases to decrease
		if (self.config.earlyStopping == true) then
			if (testErr < self.bestError) then
				self.bestError = testErr
				self.bestModel = self.module:clone()
				self.extraIndex = 1
			else
				if (self.extraIndex < self.config.extraEpochs) then
					self.extraIndex = self.extraIndex + 1
				else
					print("# Composer: stopping - you have reached the maximum number of epochs after the best model")
					print("# Composer: best error: " .. self.bestError)
					self.module = self.bestModel:clone()
					break					 
				end
			end
		end

	end
end
