require 'torch'
require 'nn'

require 'cutorch'
require 'cunn'
require 'cudnn'

require 'nngraph'
require 'optim'
require 'paths'
require 'xlua'
local sh = require 'sh' -- from https://github.com/zserge/luash


require 'composition/composer.lua'

local lua_utils = require 'utils.lua_utils'
local compose_utils = require 'utils.compose_utils'
local nonliniarities = require 'utils.nonliniarities'

require 'composition/composition_models.CompositionModel'
require 'composition/composition_models.Matrix'
require 'composition/composition_models.FullAdd'

print(_VERSION)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- command-line options
cmd = torch.CmdLine()
cmd:text()
cmd:text('gWordcomp: compositionality modelling')
cmd:text()
cmd:text('Options:')
cmd:argument('-model', 'compositionality model to train: Matrix|FullAdd')
cmd:option('-nonlinearity', 'tanh', 'nonlinearity to use, if needed by the architecture: none|tanh|sigmoid|reLU')
cmd:option('-dim', 300, 'embeddings set, chosen via dimensionality: 300')
cmd:option('-dataset', 'english_compounds_composition_dataset', 'dataset to use: english_compounds_composition_dataset')
cmd:option('-embeddings', 'glove_encow14ax_enwiki_8B.400k_l2norm_axis01', 'embeddings to use: glove_encow14ax_enwiki_8B.400k_l2norm_axis01')

cmd:option('-gpuid', 1, 'gpuid')
cmd:option('-criterion', 'mse', 'criterion to use: mse')
cmd:option('-dropout', 0.1, 'dropout')
cmd:option('-extraEpochs', 5, 'extraEpochs for early stopping')
cmd:option('-batchSize', 100, 'mini-batch size (number between 1 and the size of the training data')
cmd:option('-outputDir', 'models', 'output directory to store the trained models')
cmd:option('-manual_seed', 123, 'manual seed for repeatable experiments')
cmd:option('-testDev', true, 'test model on dev dataset')
cmd:option('-testTest', false, 'test model on test dataset')
cmd:option('-testFull', false, 'test model on full dataset')

cmd:text()

opt = cmd:parse(arg)

print('using CUDA on GPU ' .. opt.gpuid .. '...')
cutorch.setDevice(opt.gpuid)
torch.manualSeed(opt.manual_seed) 
cutorch.manualSeed(opt.manual_seed, opt.gpuid)
print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

---------------------------------------------------------------------------
---------------------------------------------------------------------------

-- config
local config = {
	rundir = paths.concat(opt.outputDir, opt.dataset, opt.embeddings, opt.dim .. 'd'),
	batchSize = opt.batchSize,
	optimizer = 'adagrad',
	criterion = opt.criterion,
	adagrad_config = {
		learningRate = 1e-2,
		learningRateDecay = 0,
		weightDecay = 0
	},
	earlyStopping = true,
	extraEpochs = opt.extraEpochs,
	manualSeed = opt.manual_seed,
	gpuid = opt.gpuid,
	dropout = opt.dropout,
	cosineNeighbours = 0
}

local tf=os.date('%Y-%m-%d_%H-%M',os.time())

-- fix seed, for repeatable experiments
torch.manualSeed(config.manualSeed)

local configname = opt.model .. '_' .. opt.nonlinearity .. '_' .. config.optimizer .. "_batch" .. config.batchSize .. "_" .. config.criterion

config.saveName = paths.concat(config.rundir, "model_" .. configname .. "_" .. tf)
xlua.log(config.saveName .. ".log")

print("==> config", config)
print("==> optimizer_config: ", config.optimizer_config)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- load data
local trainSet, devSet, testSet, fullSet = compose_utils:loadDatasets(opt.dataset, opt.minNum)
local cmhDictionary, cmhEmbeddings = compose_utils:loadCMHDense(opt.dataset, opt.embeddings, opt.dim)

local sz = cmhEmbeddings:size()[2]
---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- composition models
local composition_models = {}
local nl = {}

composition_models['Matrix'] = torch.Matrix(sz * 2, sz)
composition_models['FullAdd'] = torch.FullAdd(sz * 2, sz)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- nonlinearities
nl['tanh'] = nonliniarities:tanhNonlinearity()
nl['hardTanh'] = nonliniarities:hardTanhNonlinearity()
nl['sigmoid'] = nonliniarities:sigmoidNonlinearity()
nl['reLU'] = nonliniarities:reLUNonlinearity()
nl['none'] = nonliniarities:noneNonlinearity()
---------------------------------------------------------------------------
---------------------------------------------------------------------------

local composition_model = composition_models[opt.model]
local nonlinearity = nl[opt.nonlinearity]
local mlp = composition_model:architecture(config)
composition_model:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)

local timer = torch.Timer()
if (composition_model.isTrainable == true) then
	composition_model:train()
	print("==> Training ended");
end

torch.save(config.saveName .. ".bin", mlp);

print("==> Saving predictions...")

composition_model:predict(opt.testDev, opt.testTest, opt.testFull, cmhDictionary, devSet, testSet, fullSet)

print('Time elapsed (real): ' .. lua_utils:secondsToClock(timer:time().real))
print('Time elapsed (user): ' .. lua_utils:secondsToClock(timer:time().user))
print('Time elapsed (sys): ' .. lua_utils:secondsToClock(timer:time().sys))

print("==> Model saved under " .. config.saveName .. ".bin");

local eval = sh.command('python eval/composition_eval.py')
if (opt.testDev == true) then
	print("running evaluation on dev...")
	local score = eval(
		config.saveName .. '_dev.pred',
	 	paths.concat('data', opt.dataset, 'embeddings', opt.embeddings, opt.embeddings .. '.' .. opt.dim .. 'd_cmh.dm'),
		config.saveName .. '_dev')
	print(score)
end

if (opt.testTest == true) then
	print("running evaluation on test...")
	local score = eval(
		config.saveName .. '_test.pred',
	 	paths.concat('data', opt.dataset, 'embeddings', opt.embeddings, opt.embeddings .. '.' .. opt.dim .. 'd_cmh.dm'),
		config.saveName .. '_test')
	print(score)
end

if (opt.testFull == true) then
	print("running evaluation on the full dataset...")
	local score = eval(
		config.saveName .. '_full.pred',
	 	paths.concat('data', opt.dataset, 'embeddings', opt.embeddings, opt.embeddings .. '.' .. opt.dim .. 'd_cmh.dm'),
		config.saveName .. '_full')
	print(score)
end
