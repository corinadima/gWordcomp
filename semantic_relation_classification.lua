require 'torch'
require 'nn'
require 'nngraph'
require 'paths'

local lua_utils = require 'utils.lua_utils'
local dataLoader = require 'utils.dataLoader'
local Classifier = require 'classification.Classifier'

local stringx = require('pl.stringx')
local tablex = require('pl.tablex')

-- command-line options
cmd = torch.CmdLine()
cmd:text()
cmd:text('semantic relation classification')
cmd:text()
cmd:text('Options:')
cmd:argument('-model', 'architecture to train: basic_600x300|compoM_300x600|compoFA_300x600') -- see possible architectures below
cmd:option('-dim', 300, 'embeddings set, chosen via dimensionality: 300')
cmd:option('-dataset', 'english_oseaghdha_1443_compounds', 'dataset to use: english_oseaghdha_1443_compounds|english_tratz_19158_compounds')
cmd:option('-embeddings', 'glove_encow14ax_enwiki_8B.400k_l2norm_axis01', 'embeddings to use: glove_encow14ax_enwiki_8B.400k_l2norm_axis01')
cmd:option('-dropout', 0.1, 'dropout') 
cmd:option('-grad_clip', 5)
cmd:option('-extraEpochs', 100, 'extraEpochs for early stopping')
cmd:option('-batchSize', 100, 'mini-batch size (number between 1 and the size of the training data')
cmd:option('-outputDir', 'en_semantic_models', 'output directory to store the trained models')
cmd:option('-gpuid', 1,'which gpu to use. -1 = use CPU')
cmd:option('-manual_seed', 9, 'manual seed for repeatable experiments')

cmd:text()

opt = cmd:parse(arg)
---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- cuda setup

if opt.gpuid >= 0 then
	local ok, cunn = pcall(require, 'cunn')
	local ok2, cutorch = pcall(require, 'cutorch')
	if not ok then print('package cunn not found!') end
	if not ok2 then print('package cutorch not found!') end
	if ok and ok2 then
		print('using CUDA on GPU ' .. opt.gpuid .. '...')
		cutorch.setDevice(opt.gpuid)
		torch.manualSeed(opt.manual_seed) 
		cutorch.manualSeed(opt.manual_seed, opt.gpuid)
		print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end
------------------------------------------------------------------------------
------------------------------------------------------------------------------
-- dataset configurations

local os_dataset_config = {
	dictionary = paths.concat('data', opt.dataset, "oseaghdha.dict"),
	relations = paths.concat('data', opt.dataset, "oseaghdha_relations.txt"),
	foldsDir = paths.concat('data', opt.dataset, "5-fold"),
	foldsNamePattern = "oseaghdha_cv",
	noFolds = 5,
	delimiter = ' ',
}

os_dataset_config.mh_embeddings_path = paths.concat('data', opt.dataset, 'embeddings', opt.embeddings,
 	opt.embeddings .. '.' .. opt.dim .. 'd_os_mh.dm')

local tratz_19158_dataset_config = {
	dictionary = paths.concat('data', opt.dataset, "tratz_2011_recoded_constituent_dictionary.txt"),
	relations = paths.concat('data', opt.dataset, "tratz_2011_relations_full.txt"),
	foldsDir = paths.concat('data', opt.dataset, "10-fold"),
	foldsNamePattern = "tratz_2011_recoded_cv",
	noFolds = 10,
	delimiter = ',',
}

tratz_19158_dataset_config.mh_embeddings_path = paths.concat('data', opt.dataset, 'embeddings', opt.embeddings,
 	opt.embeddings .. '.' .. opt.dim .. 'd_t2011_mh.dm')

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-- config
local config = {
	embeddings = opt.embeddings,
	rundir = paths.concat(opt.outputDir, opt.dataset, opt.embeddings, opt.dim .. 'd'),
	batchSize = opt.batchSize,
	optimizer = 'adagrad',
	-- adagrad_config = {
	-- 	learningRate = 1e-1,
	-- 	learningRateDecay = 1e-5,
	-- 	weightDecay = 1e-5
	-- }, -- TH config
	adagrad_config = {
		learningRate = 5e-2,
		learningRateDecay = 1e-5,
		weightDecay = 1e-5
	}, -- OS config
	asgd_config = {
		eta0 = 1e-4,
		lambda = 1e-4,
		alpha = 0.75,
		t0 = 1e6
	},
	earlyStopping = true,
	extraEpochs = opt.extraEpochs,
	manualSeed = opt.manual_seed,
	gpuid = opt.gpuid,
	dropout = opt.dropout,
	cosineNeighbours = 0,
	adam_config = {
		learningRate = opt.lr
	},
	grad_clip = opt.grad_clip,
	-- dataset_config = tratz_19158_dataset_config,
	dataset_config = os_dataset_config,
}

config.pretrainedFullAdd = 'model_FullAdd_tanh_adagrad_batch100_mse_2016-05-09_18-36'
config.pretrainedMatrix = 'model_Matrix_tanh_adagrad_batch100_mse_2016-05-09_18-11'

local configname = opt.model .. '_' .. config.optimizer .. "_batch" .. config.batchSize
local tf=os.date('%Y-%m-%d_%H-%M',os.time())
config.saveName = paths.concat(config.rundir, "model_" .. configname .. "_" .. tf)
xlua.log(config.saveName .. ".log")


------------------------------------------------------------------------------
------------------------------------------------------------------------------
-- Load data
print("==> loading dictionary...")
local dictionary = dataLoader.loadDictionary(config.dataset_config.dictionary)
print("==> loading relations...")
local relations = dataLoader.loadDictionary(config.dataset_config.relations)
print("==> dictionary and relations loaded.")

print("==> loading embeddings...")
local mhDictionary, mhEmbeddings = dataLoader.loadDenseMatrix(config.dataset_config.mh_embeddings_path)
print('==> embeddings loaded, size:', mhEmbeddings:size())

print("==> loading cross-validation sets...")
local folds = {}
local textFolds = {}

local total = 0
local constituents_nf = 0

function load_classification_dataset(fileName, mhDictionary, mhEmbeddings, classes, delimiter)
	local tot = 0
	local const_nf = 0
	local dataset = {}
	local textDataset = {}
	local i = 1
	local sz = mhEmbeddings:size()[2]

	io.input(fileName)
	local lines = io.lines()
	for line in io.lines() do
		local sp = stringx.split(stringx.strip(line), delimiter)
		local c1_idx = tablex.find(mhDictionary, sp[1])
		local c2_idx = tablex.find(mhDictionary, sp[2])
		local class_idx = tablex.find(classes, sp[3])
		table.insert(textDataset, {sp[1], sp[2], sp[3]})
		if (c1_idx == nil) then
			print('=======> ' .. sp[1] .. ' not found from ' .. fileName)
			const_nf = const_nf + 1
			print(sp)
			c1_idx = tablex.find(mhDictionary, '<unk>')
			print(c1_idx)
		end
		if (c2_idx == nil) then
			print('=======> ' .. sp[2] .. ' not found from ' .. fileName)
			const_nf = const_nf + 1
			print(sp)
			c2_idx = tablex.find(mhDictionary, '<unk>')
			print(c2_idx)
		end
		if ((c1_idx ~= nil) and (c2_idx ~= nil)) then
			local input = torch.zeros(2, sz)
			input[1] = mhEmbeddings[c1_idx]:clone()
			input[2] = mhEmbeddings[c2_idx]:clone()
			local output = class_idx
			dataset[i] = {input, output}
			i = i + 1 
		end
		tot = tot + 1
	end

	return tot, const_nf, dataset, textDataset
end

-- train setup
local classes = {}
for key, value in pairs(relations) do table.insert(classes, value) end
table.remove(classes)

for i = 1, config.dataset_config.noFolds do
	local fileName = paths.concat(config.dataset_config.foldsDir, config.dataset_config.foldsNamePattern .. i .. ".txt")
	local tot, const_nf, dataset, textDataset = load_classification_dataset(fileName, mhDictionary, mhEmbeddings, classes, config.dataset_config.delimiter)
	total = total + tot
	constituents_nf = constituents_nf + const_nf
	folds[i] = dataset
	textFolds[i] = textDataset
end
print(total .. ' compounds in the dataset. ')
print(constituents_nf .. ' constituents were not found')
------------------------------------------------------------------------------
------------------------------------------------------------------------------

-- use the matrix composition as pretraining;
-- the training affects the composition part and retrains it
-- inputs: c1 and c2, trained word embeddings for the two constituents
-- W matrix size: 600x300
-- output: softmax over noRel relations
function pretrain_matrix_600x300(noInputs, noRel, gpuid, dropout_coef, config)
	local datasetDir = 'english_compounds_composition_dataset'
	local size = noInputs

	-- load pre-trained composition model
	local pretrainedPath = paths.concat('models', datasetDir, config.embeddings, size .. 'd', config.pretrainedMatrix .. '.bin')
	print("==> loading mlp from " .. pretrainedPath)
	local matrixMLP = torch.load(pretrainedPath);
	print("==> model loaded.")
	for indexNode, node in ipairs(matrixMLP.forwardnodes) do
	  if node.data.module then
	    print(node.data.module)
	  end
	end

	-----------------------------------

	local c1 = nn.Identity()();
	local c2 = nn.Identity()();
	local matrix = matrixMLP({c1, c2})

	local dropout = nn.Dropout(dropout_coef)({matrix})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({c1, c2}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp
end

-- use the fullAdd composition as pretraining;
-- the training affects the composition part and retrains it
-- inputs: c1 and c2, trained word embeddings for the two constituents
-- W matrix size: 600x300
-- output: softmax over noRel relations
function pretrain_fullAdd_600x300(noInputs, noRel, gpuid, dropout_coef, config)
	local datasetDir = 'english_compounds_composition_dataset'
	local size = noInputs

	local pretrainedPath = paths.concat('models', datasetDir, config.embeddings, size .. 'd', config.pretrainedFullAdd .. '.bin')
	print("==> loading mlp from " .. pretrainedPath)
	local fullAddMLP = torch.load(pretrainedPath);
	print("==> model loaded.")
	for indexNode, node in ipairs(fullAddMLP.forwardnodes) do
	  if node.data.module then
	    print(node.data.module)
	  end
	end

	-----------------------------------

	local c1 = nn.Identity()();
	local c2 = nn.Identity()();
	local fullAdd = fullAddMLP({c1, c2})

	local dropout = nn.Dropout(dropout_coef)({fullAdd})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({c1, c2}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end

	return mlp
end

-- use both matrix and fullAdd composition as pretraining;
-- the training affects the composition part and retrains it
-- inputs: c1 and c2, trained word embeddings for the two constituents
-- W matrix size: 2x600x300
-- output: softmax over noRel relations
function pretrain_matrix_fullAdd_600x600(noInputs, noRel, gpuid, dropout_coef)
	local datasetDir = 'english_compounds_composition_dataset'
	local size = noInputs

	local pretrainedPath = paths.concat('models', datasetDir, config.embeddings, size .. 'd', config.pretrainedFullAdd.. '.bin')
	print("==> loading mlp from " .. pretrainedPath)
	local fullAddMLP = torch.load(pretrainedPath)
	print("==> model loaded.")
	for indexNode, node in ipairs(fullAddMLP.forwardnodes) do
	  if node.data.module then
	    print(node.data.module)
	  end
	end

	pretrainedPath = paths.concat('models', datasetDir, config.embeddings, size .. 'd', config.pretrainedMatrix.. '.bin')
	print("==> loading mlp from " .. pretrainedPath)
	local matrixMLP = torch.load(pretrainedPath)

	print("==> model loaded.")
	for indexNode, node in ipairs(matrixMLP.forwardnodes) do
	  if node.data.module then
	    print(node.data.module)
	  end
	end

	-----------------------------------

	local c1 = nn.Identity()();
	local c2 = nn.Identity()();
	local fullAdd = fullAddMLP({c1, c2})
	local matrix = matrixMLP({c1, c2})

	local join = nn.JoinTable(2)({fullAdd, matrix})

	local dropout = nn.Dropout(dropout_coef)({join})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs * 2, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({c1, c2}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end

	return mlp

end


-- use the matrix composition as pretraining, 
-- but keep it fixed by using as input directly the composed representation
-- (training does not affect the composition part)
function compoM_300x600(noInputs, noRel, gpuid, dropout_coef, config)
	local datasetDir = 'english_compounds_composition_dataset'
	local size = noInputs

	-- load pre-trained composition model
	local pretrainedPath = paths.concat('models', datasetDir, config.embeddings, size .. 'd', config.pretrainedMatrix .. '.bin')
	print("==> loading mlp from " .. pretrainedPath)
	local matrixMLP = torch.load(pretrainedPath);
	print("==> model loaded.")

	-----------------------------------

	local compo = nn.Identity()(); -- expects composed representation as input

	local W = nn.Linear(noInputs, noInputs * 2)({compo})
	local dropout = nn.Dropout(dropout_coef)({W})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs * 2, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({compo}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp, matrixMLP
end

-- use the full add composition as pretraining, 
-- but keep it fixed by using as input directly the composed representation
-- (training does not affect the composition part)
function compoFA_300x600(noInputs, noRel, gpuid, dropout_coef, config)
	local datasetDir = 'english_compounds_composition_dataset'
	local size = noInputs

	-- load pre-trained composition model
	local pretrainedPath = paths.concat('models', datasetDir, config.embeddings, size .. 'd', config.pretrainedFullAdd .. '.bin')
	print("==> loading mlp from " .. pretrainedPath)
	local fullAddMLP = torch.load(pretrainedPath);
	print("==> model loaded.")

	-----------------------------------

	local compo = nn.Identity()(); -- expects composed representation as input

	local W = nn.Linear(noInputs, noInputs * 2)({compo})
	local dropout = nn.Dropout(dropout_coef)({W})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs * 2, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({compo}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp, fullAddMLP
end

-- use the composed vector as an additional, fixed input 
-- inputs: c1, c2 (the original word representations), compo (the composed representation, using the pretrained matrix composition)
-- W matrix size: 900x900
function c1c2_compoM_900x900(noInputs, noRel, gpuid, dropout_coef, config)
	local datasetDir = 'english_compounds_composition_dataset'
	local size = noInputs

	-- load pre-trained composition model
	local pretrainedPath = paths.concat('models', datasetDir, config.embeddings, size .. 'd', config.pretrainedMatrix .. '.bin')
	print("==> loading mlp from " .. pretrainedPath)
	local matrixMLP = torch.load(pretrainedPath);
	print("==> model loaded.")

	-----------------------------------

	-- expects initial representations as well as composed representations as input
	local c1 = nn.Identity()(); 
	local c2 = nn.Identity()(); 
	local compo = nn.Identity()(); 

	local join = nn.JoinTable(2)({c1, c2, compo})
	local reshape = nn.Reshape(noInputs * 3, true)({join})
	local W = nn.Linear(noInputs * 3, noInputs * 3)({reshape})
	local dropout = nn.Dropout(dropout_coef)({W})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs * 3, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({c1, c2, compo}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp, matrixMLP
end

-- use the composed vector as an additional, fixed input 
-- inputs: c1, c2 (the original word representations), compo (the composed representation, using the pretrained full additive composition)
-- W matrix size: 900x900
function c1c2_compoFA_900x900(noInputs, noRel, gpuid, dropout_coef, config)
	local datasetDir = 'english_compounds_composition_dataset'
	local size = noInputs

	-- load pre-trained composition model
	local pretrainedPath = paths.concat('models', datasetDir, config.embeddings, size .. 'd', config.pretrainedFullAdd .. '.bin')
	print("==> loading mlp from " .. pretrainedPath)
	local fullAddMLP = torch.load(pretrainedPath);
	print("==> model loaded.")

	-----------------------------------

	-- expects initial representations as well as composed representations as input
	local c1 = nn.Identity()(); 
	local c2 = nn.Identity()(); 
	local compo = nn.Identity()(); 

	local join = nn.JoinTable(2)({c1, c2, compo})
	local reshape = nn.Reshape(noInputs * 3, true)({join})
	local W = nn.Linear(noInputs * 3, noInputs * 3)({reshape})
	local dropout = nn.Dropout(dropout_coef)({W})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs * 3, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({c1, c2, compo}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp, fullAddMLP
end


function c1c2_compoMcompoFA_1200x1200(noInputs, noRel, gpuid, dropout_coef, config)
	local datasetDir = 'english_compounds_composition_dataset'
	local size = noInputs

	-- load pre-trained composition model
	local pretrainedPath = paths.concat('models', datasetDir, config.embeddings, size .. 'd', config.pretrainedFullAdd .. '.bin')
	print("==> loading mlp from " .. pretrainedPath)
	local fullAddMLP = torch.load(pretrainedPath);
	print("==> model 1 loaded.")

	pretrainedPath = paths.concat('models', datasetDir, config.embeddings, size .. 'd', config.pretrainedMatrix .. '.bin')
	print("==> loading mlp from " .. pretrainedPath)
	local matrixMLP = torch.load(pretrainedPath);
	print("==> model 2 loaded.")

	-----------------------------------

	-- expects initial representations as well as composed representations as input
	local c1 = nn.Identity()(); 
	local c2 = nn.Identity()(); 
	local compoM = nn.Identity()(); 
	local compoFA = nn.Identity()(); 

	local join = nn.JoinTable(2)({c1, c2, compoM, compoFA})
	local reshape = nn.Reshape(noInputs * 4, true)({join})
	local W = nn.Linear(noInputs * 4, noInputs * 4)({reshape})
	local dropout = nn.Dropout(dropout_coef)({W})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs * 4, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({c1, c2, compoM, compoFA}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp, matrixMLP, fullAddMLP
end

-- basic models

function basic_600x300(noInputs, noRel, gpuid, dropout_coef)
	local c1 = nn.Identity()();
	local c2 = nn.Identity()();

	local join = nn.JoinTable(2)({c1, c2})
	local reshape = nn.Reshape(noInputs * 2, true)({join})
	local W = nn.Linear(noInputs * 2, noInputs)({reshape})

	local dropout = nn.Dropout(dropout_coef)({W})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({c1, c2}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp
end

function basic_600x600(noInputs, noRel, gpuid, dropout_coef)

	local c1 = nn.Identity()();
	local c2 = nn.Identity()();

	local join = nn.JoinTable(2)({c1, c2})
	local reshape = nn.Reshape(noInputs * 2, true)({join})
	local W = nn.Linear(noInputs * 2, noInputs * 2)({reshape})

	local dropout = nn.Dropout(dropout_coef)({W})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs * 2, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({c1, c2}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end

	return mlp
end

function basic_600x1200(noInputs, noRel, gpuid, dropout_coef)

	local c1 = nn.Identity()();
	local c2 = nn.Identity()();

	local join = nn.JoinTable(2)({c1, c2})
	local reshape = nn.Reshape(noInputs * 2, true)({join})
	local W = nn.Linear(noInputs * 2, noInputs * 4)({reshape})

	local dropout = nn.Dropout(dropout_coef)({W})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs * 4, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({c1, c2}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end

	return mlp
end

function basic_600x2400(noInputs, noRel, gpuid, dropout_coef)

	local c1 = nn.Identity()();
	local c2 = nn.Identity()();

	local join = nn.JoinTable(2)({c1, c2})
	local reshape = nn.Reshape(noInputs * 2, true)({join})
	local W = nn.Linear(noInputs * 2, noInputs * 6)({reshape})

	local dropout = nn.Dropout(dropout_coef)({W})
	local nonlin = nn.ReLU()({dropout})

	local W_rel = nn.Linear(noInputs * 6, noRel)({nonlin})
	nonlin = nn.Tanh()({W_rel})
	local softmax = nn.LogSoftMax()({nonlin})

	local mlp = nn.gModule({c1, c2}, {softmax})
	if gpuid >= 0 then
		mlp:cuda()
	end

	return mlp
end


-- ------------------------------------------------------------------------------
-- ------------------------------------------------------------------------------
-- -- training

function save_predictions(textFold, predictions, classes, saveName)
	local outputFileName = saveName .. '.pred'
	local f = io.open(outputFileName, "w")

	for i = 1, predictions:size()[1] do
		f:write(textFold[i][1] .. "," .. textFold[i][2] .. "," .. textFold[i][3] .. "," .. classes[predictions[i][1]] .. "\n")
	end

	print("==> Predictions saved under " .. saveName .. '.pred')

	f:close()
end

function cv_train(folds, textFolds, size, classes, gpuid, dropout_coef, config, model)
	local f1_scores = torch.Tensor(#folds)
	local accuracies = torch.Tensor(#folds)
	local corrects = torch.Tensor(#folds)
	local totals = torch.Tensor(#folds)

	local bestModel = nil
	local foldPredictions = nil

	for i = 1, #folds do
		local cv_trainDataset = {}
		local cv_testDataset = folds[i]
		local idx = 0
		for j = 1, #folds do
			if i ~= j then
				for k = 1, #folds[j] do
					idx = idx + 1
					cv_trainDataset[idx] = folds[j][k]
				end
			end
		end

		print('****** Fold ' .. i .. " ********")
		local tot = tablex.size(cv_trainDataset) + tablex.size(cv_testDataset)
		print("Train size: " .. tablex.size(cv_trainDataset) .. "; test size: " .. tablex.size(cv_testDataset) .. "; total: " .. tot)

		local classifier = Classifier:init(size, classes, nil, nil)

		local mlp = nil

		-- basic configurations
		if (model == 'basic_600x300') then
			mlp = basic_600x300(size, tablex.size(classes), gpuid, dropout_coef, config)
		elseif (model == 'basic_600x600') then 
			mlp = basic_600x600(size, tablex.size(classes), gpuid, dropout_coef, config)
		elseif (model == 'basic_600x1200') then
			mlp = basic_600x1200(size, tablex.size(classes), gpuid, dropout_coef, config)
		elseif (model == 'basic_600x2400') then
			mlp = basic_600x2400(size, tablex.size(classes), gpuid, dropout_coef, config)

		-- pre-trained & fine-tuned configurations
		elseif (model == 'pretrain_matrix_600x300') then
			mlp = pretrain_matrix_600x300(size, tablex.size(classes), gpuid, dropout_coef, config)
		elseif (model == 'pretrain_fullAdd_600x300') then
			mlp = pretrain_fullAdd_600x300(size, tablex.size(classes), gpuid, dropout_coef, config)
		elseif (model == 'pretrain_matrix_fullAdd_600x600') then -- was combi
			mlp = pretrain_matrix_fullAdd_600x600(size, tablex.size(classes), gpuid, dropout_coef, config)

		-- pre-trained but not fine-tuned configurations
		elseif (model == 'compoM_300x600') then
			mlp, classifier.pretrainedModel = compoM_300x600(size, tablex.size(classes), gpuid, dropout_coef, config)
		elseif (model == 'compoFA_300x600') then
			mlp, classifier.pretrainedModel = compoFA_300x600(size, tablex.size(classes), gpuid, dropout_coef, config)

		elseif (model == 'c1c2_compoM_900x900') then
			classifier.dualInput = true
			mlp, classifier.pretrainedModel = c1c2_compoM_900x900(size, tablex.size(classes), gpuid, dropout_coef, config)
		elseif (model == 'c1c2_compoFA_900x900') then
			classifier.dualInput = true
			mlp, classifier.pretrainedModel = c1c2_compoFA_900x900(size, tablex.size(classes), gpuid, dropout_coef, config)
		elseif (model == 'c1c2_compoMcompoFA_1200x1200') then
			classifier.dualInput = true
			classifier.quadrupleInput = true
			mlp, classifier.pretrainedModel, classifier.pretrainedModel2 = c1c2_compoMcompoFA_1200x1200(size, tablex.size(classes), gpuid, dropout_coef, config)

		else
			error('Unknown model')
		end

		classifier:architecture(mlp, config)
		classifier:data(cv_trainDataset, cv_testDataset)
		accuracies[i], bestModel, foldPredictions = classifier:train()

		local foldSaveName = config.saveName .. "_" .. "fold_" .. i
		torch.save(foldSaveName .. ".bin", bestModel);
		print("==> Model saved under " .. foldSaveName .. ".bin");
		save_predictions(textFolds[i], foldPredictions, classes, foldSaveName)
	end
	for i = 1, #folds do
		print("Fold " .. i .. " " .. accuracies[i])
	end
	print("Mean accuracy " .. torch.mean(accuracies))
	print(opt.dropout)
end

cv_train(folds, textFolds, opt.dim, classes, opt.gpuid, opt.dropout, config, opt.model)



