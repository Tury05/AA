using DelimitedFiles;
using Statistics;
using Flux;
using Flux.Losses;
using Random;
using Plots;
using ScikitLearn;

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

# Funciones para generar una BD con imagenes

using JLD2;
using Images;

# Functions that allow the conversion from images to Float64 arrays
imageToGrayArray(image:: Array{RGB{Normed{UInt8,8}},2}) = convert(Array{Float64,2}, gray.(Gray.(image)));
imageToGrayArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToGrayArray(RGB.(image));
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float32, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

# Function to read all of the images in a folder and return them as 2 Float64 arrays: one with color components (3D array) and the other with grayscale components (2D array)
function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            # Check that they are color images
            @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            # Add the image to the vector of images
            push!(images, image);
        end;
    end;
    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
    return (imageToColorArray.(images), imageToGrayArray.(images));
end;

# Functions to load the dataset
function loadTrainingDataset(positiveFolderName::String, negativeFolderName::String)
    (positivesColor, positivesGray) = loadFolderImages(positiveFolderName);
    (negativesColor, negativesGray) = loadFolderImages(negativeFolderName);
    targets = [trues(length(positivesColor)); falses(length(negativesColor))];
    return ([positivesColor; negativesColor], [positivesGray; negativesGray], targets);
end;
loadTestDataset() = ((colorMatrix,_) = loadFolderImages("test"); return colorMatrix; );

function loadImage(fileName::String)
	if  any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"])
        image = load(fileName);
        # Check that they are color images
        @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))

        return (imageToColorArray(image), imageToGrayArray(image));
    else
    	throw("No es una imagen");
    end;
end;

function imageToData(fileName::String)

	image,_ = loadImage(fileName);

	data = Array{Float64, 1}(undef, 9);

	fil = convert(Int,round(size(image,1)/7));
	col = convert(Int,round(size(image,2)/3));

	k = 1
	for j in 1:3
		for c in 1:3
			data[k] = mean(image[(fil*(j*2-1)):(fil*(j*2)),col:(col*2),c]);
			k += 1;
		end;
	end;

	return data;
end;

function santaImagesToDatasets(santaFolder::String, notSantaFolder::String)

	santaImages,_ = loadFolderImages(santaFolder);
	notSantaImages,_ = loadFolderImages(notSantaFolder);
	images = (santaImages, notSantaImages);

	santaDataset = Array{Float64, 2}(undef, size(santaImages,1), 9);
	notSantaDataset = Array{Float64, 2}(undef, size(notSantaImages,1), 9);
	datasets = (santaDataset, notSantaDataset);

	for s in 1:2
		i = 1
		for colorMatrix in images[s]

			fil = convert(Int,round(size(colorMatrix,1)/7));
			col = convert(Int,round(size(colorMatrix,2)/3));

			k = 1
			for j in 1:3
				for c in 1:3
					datasets[s][i,k] = mean(colorMatrix[(fil*(j*2-1)):(fil*(j*2)),col:(col*2),c]);
					k += 1;
				end;
			end;

			i += 1;
		end;
	end;

	return datasets;
end;


function santaImagesToDatasets2(santaFolder::String, notSantaFolder::String)

	santaDataset2 = Array{Float64, 2}(undef, size(datasets[1],1), 7);
	notsantaDataset2 = Array{Float64, 2}(undef, size(datasets[2],1), 7);

	datasets = santaImagesToDatasets(santaFolder, notSantaFolder);
	
	for s in 1:2
		for i in datasets[s]
			conj1 = (datasets[s][i,1], datasets[s][i,2], datasets[s][i,3]);
			conj2 = (datasets[s][i,7], datasets[s][i,8], datasets[s][i,9]);
			datasets2[s][i, 1] = mean(conj1);
			datasets2[s][i, 2] = std(conj2);
			datasets2[s][i, 3] = datasets[s][i,4];
			datasets2[s][i, 4] = datasets[s][i,5];
			datasets2[s][i, 5] = datasets[s][i,6];
			datasets2[s][i, 6] = mean(conj2);
			datasets2[s][i, 7] = std(conj2);
		end;
	end;
	
	return datasets2;
end;
			


function randDataset(a1::AbstractArray{Float64,2}, a2::AbstractArray{Float64,2})
	inDSLength = size(a1,1)+size(a2,1);
	perm = randperm(inDSLength);
	inDS = Array{Float64, 2}(undef, inDSLength, 9);
	outDS = Array{Bool}(undef, inDSLength);
	k = 1;

	for i in 1:size(a1,1)
		inDS[perm[k],:] = a1[i,:];
		outDS[perm[k]] = true;
		k += 1;
	end;

	for i in 1:size(a2,1)
		inDS[perm[k],:] = a2[i,:];
		outDS[perm[k]] = false;
		k += 1;
	end;

	return (inDS, outDS);
end;



maxMinNorm = function (v, min, max)
	return (v .- min)./(max .- min);
end;

media0Norm = function (v, avg, std)
	return (v .- avg)./std;
end;

function readData(dataset)
	data = readdlm(dataset, ',')
	f, c = size(data);

	inDS = data[:, 1:c-1];
	outDS = data[:, c];
	return inDS, outDS
end

#1
function oneHotEncoding(feature::AbstractArray{<:Any,1},classes::AbstractArray{<:Any,1}=[])
	if length(classes) == 2
		out = classes[1] .== feature
	else
		out = Array{Bool, 2}(undef, length(feature), length(classes))
		for i in 1:size(out, 1)
			out[i, :] = feature[i].==classes
		end
	end
	return out
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

function oneHotEncoding(feature::AbstractArray{<:Bool,1})
	m = reshape(feature, :, 1)
	return m
end;


#2
calculateMinMaxNormalizationParameters = function (inputs::AbstractArray{<:Real,2})
	(minimum(inputs, dims=1), maximum(inputs, dims=1))
end

calculateZeroMeanNormalizationParameters = function (inputs::AbstractArray{<:Real,2})
	(mean(inputs, dims=1), std(inputs, dims=1))
end


function normalizeMinMax!(inputs::AbstractArray{Float32,2},
	minMax::NTuple{2, AbstractArray{<:Real,2}})
	for i in 1:size(inputs,2)
		inputs[:, i] = maxMinNorm(inputs[:, i], minMax[1][i], minMax[2][i])
	end
end

function normalizeMinMax!(inputs::AbstractArray{Float32,2})
	minMax = calculateMinMaxNormalizationParameters(inputs)
	for i in 1:size(inputs,2)
		inputs[:, i] = maxMinNorm(inputs[:, i], minMax[1][i], minMax[2][i])
	end
end

function normalizeMinMax(inputs::AbstractArray{Float32,2},
	minMax::NTuple{2, AbstractArray{<:Real,2}}=())
	out = copy(inputs)
	for i in 1:size(out,2)
		out[:, i] = maxMinNorm(out[:, i], minMax[1][i], minMax[2][i])
	end
	return out
end

function normalizeMinMax(inputs::AbstractArray{Float32,2})
	out = copy(inputs)
	minMax = calculateMinMaxNormalizationParameters(out)
	for i in 1:size(out,2)
		out[:, i] = maxMinNorm(out[:, i], minMax[1][i], minMax[2][i])
	end
	return out
end


function normalizeZeroMean!(inputs::AbstractArray{Float32,2},
		meanStd::NTuple{2, AbstractArray{<:Real,2}})
	for i in 1:size(inputs,2)
		inputs[:, i] = media0Norm(inputs[:, i], meanStd[1][i], meanStd[2][i])
	end
end

function normalizeZeroMean!(inputs::AbstractArray{Float32,2})
	meanStd = calculateZeroMeanNormalizationParameters(inputs)
	for i in 1:size(inputs,2)
		inputs[:, i] = media0Norm(inputs[:, i], meanStd[1][i], meanStd[2][i])
	end
end

function normalizeZeroMean(inputs::AbstractArray{Float32,2},
		meanStd::NTuple{2, AbstractArray{<:Real,2}})
	out = copy(inputs)
	for i in 1:size(out,2)
		out[:, i] = media0Norm(out[:, i], meanStd[1][i], meanStd[2][i])
	end
	return out
end

function normalizeZeroMean(inputs::AbstractArray{Float32,2})
	out = copy(inputs)
	meanStd = calculateZeroMeanNormalizationParameters(out)
	for i in 1:size(out,2)
		out[:, i] = media0Norm(out[:, i], meanStd[1][i], meanStd[2][i])
	end
	return out
end

#3 (dificultad media)
function classifyOutputs(outputs::AbstractArray{<:Real,2}, threshold = 0.5)
	if size(outputs, 2) == 1
		out = outputs .>= threshold
	else
		out = falses(size(outputs))
		(_,indicesMaxEachInstance) = findmax(outputs, dims=2)
		out[indicesMaxEachInstance] .= true
	end
	return out
end


#4 (dificultad media) P�gina 11
function accuracy(target::AbstractArray{Bool,1},
		outputs::AbstractArray{Bool,1})
	@assert size(target) == size(outputs)
	classComparison = target .== outputs
	accuracy = mean(classComparison)
end

function accuracy(target::AbstractArray{Bool,2},
		outputs::AbstractArray{Bool,2})
	@assert size(target) == size(outputs)
	if size(outputs, 2) == 1
		accuracy(reshape(target, size(target, 1)), reshape(outputs, size(outputs, 1)))
	elseif size(outputs, 2) > 2
		classComparison = target .== outputs
		correctClassifications = all(classComparison, dims=2)
		accuracy = mean(correctClassifications)
	end
end

function accuracy(target::AbstractArray{Bool,1},
		outputs::AbstractArray{<:Real,1}, threshold = 0.5)
	@assert size(target) == size(outputs)
	out = outputs .>= threshold
	accuracy(target, out)
end

function accuracy(target::AbstractArray{Bool,2},
		outputs::AbstractArray{<:Real,2}, threshold = 0.5)
	@assert size(target) == size(outputs)
	if size(outputs, 2) == 1
		accuracy(reshape(target, size(target, 1)), reshape(outputs, size(outputs, 1)))
	elseif size(outputs, 2) > 2
		classifiedOut= classifyOutputs(outputs, threshold)
		accuracy(target, classifiedOut)
	end
end


#5 (dificultad alta)
function classRNA(nEntradas::Int, nSalidas::Int,
		topology::AbstractArray{<:Int,1},
		funActivacion::Any=[])
	if isempty(topology)
		throw(ArgumentError("invalid Array dimensions"))
	end;
	if isempty(funActivacion)
		funActivacion = repeat([σ], length(topology))
	end;
	if length(topology) != length(funActivacion)
		throw(ArgumentError("topology and funActivacion must have the same length"))
	end;

	ann = Chain();

	numInputsLayer = nEntradas;

	for i in 1:length(topology)
		ann = Chain(ann..., Dense(numInputsLayer, topology[i], funActivacion[i]) );
		numInputsLayer = topology[i];
	end;

	if nSalidas == 2
		Chain(ann..., Dense(numInputsLayer, 1, σ))
	elseif nSalidas > 2
		ann = Chain(ann..., Dense(numInputsLayer, nSalidas, identity))
		Chain(ann..., softmax)
	else
		throw(ArgumentError("nSalidas must be >= 2"))
	end;
end;

function entrenarClassRNA(topology::AbstractArray{<:Int,1},
		dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
		maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
	nSalidas = size(last(dataset), 2);
	nSalidas = if nSalidas == 1 2 else nSalidas end;
	ann = classRNA(size((first(dataset)), 2), nSalidas, topology);

	e = 0;
	l = Inf;
	losses = [];
	
	loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
	
	while e < maxEpochs && l > minLoss
		Flux.train!(loss, params(ann), [(first(dataset)', last(dataset)')], ADAM(learningRate));
		l = loss(first(dataset)', last(dataset)');
		e += 1;
		push!(losses, l);
	end;

	return (ann, losses);
	
end;

function entrenarClassRNA(topology::AbstractArray{<:Int,1},
		dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
		maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
	
	return entrenarClassRNA(topology, (first(dataset), reshape(last(dataset), :, 1)), maxEpochs, minLoss, learningRate);
end;


#'#6 (probar las funciones)
#
#a = [1 2; 3 4; 5 6; 7 8]

#b = [true false false; false true false; false false true; true false false]
#
#ann = entrenarClassRNA([2,2], (a,b), 10)
#
#a = rand(8,2)
#
#b = Array{Bool,1}(rand(8) .> 0.5)
#
#ann = entrenarClassRNA([2,2], (a,b), 10)'

# PRACTICA 3

# 1
function holdOut(N::Int, P::Real)
	index = randperm(N);
	ntest = floor(Int,N*P);
	index[1:ntest], index[ntest:end];
end;

function holdOut(N::Int, Ptest::Real, Pval::Real)
	itrain, itest = holdOut(N, Ptest);
	nval = floor(Int, N*Pval);
	itrain[nval:end], itest, itrain[1:nval];
end;

function subArray(dataset::AbstractArray{<:Float32,2},
	 			indexes::Array{Int64,1})
	subArr = Array{Float32, 2}(undef, size(indexes, 1), size(dataset, 2))
	for i in 1:length(indexes)
		subArr[i, :] = dataset[indexes[i], :]
	end
	return subArr
end;

#2
function entrenarClassRNA(topology::AbstractArray{<:Int,1},
		dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
		testset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
		validset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
		maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01,
		maxEpochsVal::Int=20)

	inputs = dataset[1]';
	outputs = dataset[2]';

	nSalidas = size(outputs, 1);
	nSalidas = if nSalidas == 1 2 else nSalidas end;
	ann = classRNA(size(inputs, 1), nSalidas, topology);

	e = ev = 0;
	ltrain = ltest = lvalid = lprev = Inf;
	lossestrain = lossestest = lossesvalid = [];
	bestRNA = deepcopy(ann);
	
	loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

	while e < maxEpochs && ltrain > minLoss && ev < maxEpochsVal
		Flux.train!(loss, params(ann), [(inputs, outputs)], ADAM(learningRate));

		ltrain = loss(inputs, outputs);
		ltest = loss(testset[1]', testset[2]');
		lvalid = loss(validset[1]', validset[2]');

		push!(lossestrain, ltrain);
		push!(lossestest, ltest);
		push!(lossesvalid, lvalid);
	
		if lvalid > lprev
			ev += 1
		else
			bestRNA = deepcopy(ann);
		end;

		lprev = lvalid;
		e += 1;
	end;

	return (bestRNA, lossestrain, lossestest, lossesvalid);
	
end;

function entrenarClassRNA(topology::AbstractArray{<:Int,1},
		dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
		testset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
		validset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
		maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01,
		maxEpochsVal::Int=20)
	
	return entrenarClassRNA(topology,
		(first(dataset), reshape(last(dataset), :, 1)),
		(first(testset), reshape(last(testset), :, 1)),
		(first(validset), reshape(last(validset), :, 1)),
		maxEpochs, minLoss, learningRate, maxEpochsVal);
end

########### PRUEBA ENTRENAMIENTO RNA ################
#<<<<<<< HEAD
#inDS, outDS = readData("./BBDD/iris/iris.data")
#inDS = convert(Array{Float32, 2}, inDS)
#normalizeMinMax!(inDS)
#outDS = oneHotEncoding(outDS)
#mi_red = entrenarClassRNA([8, 16, 8], (inDS, outDS))
#trained_chain = mi_red[1]
#prueba = trained_chain([a; b; c; d])
#result = classifyOutputs(transpose(prueba))
#-----Creamos datasets de entrenamiento, test y validacion
#dataset = cat(inDS, outDS, dims = 2);
#indexTrain, indexTest, indexValid = holdOut(size(inDS, 1), 0.7, 0.1);
#trainDS = subArray(dataset, indexTrain);
#testDS = subArray(dataset, indexTest);
#validDS = subArray(dataset, indexValid);
#inTrain = trainDS[:, 1:size(trainDS, 2)-3];
#outTrain = trainDS[:, size(trainDS, 2)-2:size(trainDS, 2)];
#outTrain = convert(Array{Bool, 2}, outTrain);
#inTest = testDS[:, 1:size(testDS, 2)-3];
#outTest = testDS[:, size(testDS, 2)-2:size(testDS, 2)];
#outTest = convert(Array{Bool, 2}, outTest);
#inValid = testDS[:, 1:size(testDS, 2)-3];
#outValid = validDS[:, size(validDS, 2)-2:size(validDS, 2)];
#outValid = convert(Array{Bool, 2}, outValid);
#------Entrenamos red neuronal--------
#mi_red = entrenarClassRNA([8, 16, 8], (inTrain, outTrain), (inTest), (outTest), (inValid, outValid));
#trained_chain = mi_red[1];
#prueba = trained_chain([a; b; c; d]);
#result = classifyOutputs(transpose(prueba));
#####################################################

# PRACTICA 4.1

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

	vaux = outputs .== targets;
	vp = 0; vn = 0; fp = 0; fn = 0;
	
	verd = findall(vaux)
	pos = findall(targets)
	
	vp = length(findall(vaux .& targets));
	vn = length(verd) - vp;
	fp = length(pos) - vp;
	fn = length(vaux) - (vp+vn+fp);
	
	accuracy = (vn+vp)/(vn+vp+fn+fp);
	error_rate = (fn+fp)/(vn+vp+fn+fp);
	sensitivity = vp/(fn+vp);
	specificity = vn/(vn+fp);
	pos_pred_val= vp/(vp+fp);
	neg_pred_val= vn/(vn+fn);
	F1score = 2*(sensitivity * pos_pred_val) / (sensitivity + pos_pred_val);
	confM = [vn fp; fn vp];
	
	return (accuracy, error_rate, sensitivity, specificity, pos_pred_val, neg_pred_val, F1score, confM)
end;
			
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{<:Real,1}, umbral::Real)

	vaux1 = collect(Bool, outputs .> umbral);
	vaux2 = collect(Bool, targets .> umbral);

	confusionMatrix(vaux1, vaux2);
	
end;
	

# PRACTICA 4.2

#numClasses = size(outDS, 2);
#numInstances =size(outDS, 1);
#rep = true;
#outputs = Array{Float32,2}(undef, numInstances, numClasses);


#while rep

#	for numClass in 1:numClasses
#		model,_ = entrenarClassRNA([8,16,8], (inDS, outDS[:, numClass]));
#		global outputs[:, numClass] = model(inDS');
#	end;

#	vmax = maximum(outputs, dims=2);
#	global outputs = (outputs .== vmax);
#	print(sum(unique(outputs, dims=1), dims=1));
#	global rep = any(sum(unique(outputs, dims=1), dims=1) .!= 1);
#	print(rep)
#end;


function confusionMatrix(outputs::AbstractArray{Bool,2},
	targets::AbstractArray{Bool,2}, weighted::Bool)
	
	@assert(all([in(output, unique(targets)) for output in outputs]));

	numClasses = size(targets, 2);
	numInstances =size(targets, 1);

	if numClasses == 1
		return confusionMatrix(outputs[:,1], target[:,1]);
	end;

	matrix = zeros(numClasses, numClasses);

	sensibilidades = Array{Float32, 1}(undef, numClasses);
	especificidades = Array{Float32, 1}(undef, numClasses);
	VPPs = Array{Float32, 1}(undef, numClasses);
	VPNs  = Array{Float32, 1}(undef, numClasses);
	F1s = Array{Float32, 1}(undef, numClasses);

	for i in 1:numClasses
		_,_,sensibilidades[i],especificidades[i],VPPs[i],VPNs[i],F1s[i],_ =
			confusionMatrix(outputs[:,i], targets[:,i]);
	end;

	for i in 1:numInstances
		y = findfirst(outputs[i,:]);
		x = findfirst(targets[i,:]);

		matrix[y,x] += 1;
	end

	precision = accuracy(targets, outputs);

	if weighted
		ponderacion = mapslices(r -> count(r)/numInstances, targets, dims=1);

		return precision,
			1-precision,
			mean(sensibilidades.*ponderacion),
			mean(especificidades.*ponderacion),
			mean(VPPs.*ponderacion),
			mean(VPNs.*ponderacion),
			mean(F1s.*ponderacion),
			matrix;
	else
		return precision,
			1-precision,
			mean(sensibilidades),
			mean(especificidades),
			mean(VPPs),
			mean(VPNs),
			mean(F1s),
			matrix;
	end;
end;



#Practica 5

function crossvalidation(numPatrones::Int, numConj::Int)

	vConj = collect(1:numConj)
	val = ceil(Int64, numPatrones/numConj)
	vPatrones = repeat(vConj,val)
	vPatrones = getindex(vPatrones, 1:numPatrones)
	Random.shuffle!(vPatrones)
end;


function crossvalidation(targets::AbstractArray{Bool,2}, subconj::Int)

	m,n = size(targets)
	indices = zeros(Int, m)
	
	for n in eachcol(targets)
		indices[n] = crossvalidation(sum(n), subconj)
	end
	
	return indices
end;
	
	
function crossvalidation(targets::AbstractArray{<:Any, 1}, subconj::Int)

	targets2 = oneHotEncoding(targets)
	crossvalidation(targets2, subconj)
end;



#Practica 6

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::Array{Float64,2}, targets::Array{<:Any,1}, numFolds::Int64)
    @assert(size(inputs,1)==length(targets));
    Random.seed!(0);
    
    crossValidationIndices = crossvalidation(size(inputs,1), numFolds);
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1 = Array{Float64,1}(undef, numFolds);

    for numFold in 1:numFolds
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)
            trainingInputs = inputs[crossValidationIndices.!=numFold,:];
            testInputs = inputs[crossValidationIndices.==numFold,:];
            trainingTargets = targets[crossValidationIndices.!=numFold,:];
            testTargets = targets[crossValidationIndices.==numFold,:];
        
            # Seleccionamos algoritmo
            if modelType==:SVM
                model = SVC(
                    kernel = modelHyperparameters["kernel"],
                    degree = modelHyperparameters["kernelDegree"],
                    gamma = modelHyperparameters["kernelGamma"],
                    C = modelHyperparameters["C"]
                );
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            trainingTargets = (size(trainingTargets,2) == 1) ? trainingTargets[:,1] : trainingTargets;
            model = fit!(model, trainingInputs, trainingTargets);
            testOutputs = predict(model, testInputs);
            testTargets = (size(testTargets,2) == 1) ? testTargets[:,1] : testTargets;
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);
        else
            @assert(modelType==:ANN);

            classes = unique(targets);
		    if !modelHyperparameters["normalized"]
		    	targets = oneHotEncoding(targets, classes);
		    end;

            trainingInputs = inputs[crossValidationIndices.!=numFold,:];
            testInputs = inputs[crossValidationIndices.==numFold,:];
            trainingTargets = targets[crossValidationIndices.!=numFold,:];
            testTargets = targets[crossValidationIndices.==numFold,:];
            
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            
            for numTraining in 1:modelHyperparameters["numExecutions"]
                if modelHyperparameters["validationRatio"]>0
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    ann = entrenarClassRNA(modelHyperparameters["topology"], (trainingInputs[trainingIndices,:], trainingTargets[trainingIndices,:]),
                    	(testInputs, testTargets), (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                    	modelHyperparameters["maxEpochs"], modelHyperparameters["minLoss"], modelHyperparameters["learningRate"], modelHyperparameters["maxEpochsVal"])[1];
                else
                    ann = entrenarClassRNA(modelHyperparameters["topology"], (trainingInputs, trainingTargets),
                    	modelHyperparameters["maxEpochs"], modelHyperparameters["minLoss"], modelHyperparameters["learningRate"])[1];
                end;

                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _,
                		testF1EachRepetition[numTraining], _) = if !modelHyperparameters["normalized"]

                		confusionMatrix(testTargets, ann(testInputs')');
	                else
	                	confusionMatrix(collect(Int, testTargets)[:,1], ann(testInputs')[1,:], modelHyperparameters["umbral"]);
			    	end;
                
            end;

            acc = mean(testAccuraciesEachRepetition);
            F1 = mean(testF1EachRepetition);
        end;

        testAccuracies[numFold] = acc;
        testF1[numFold] = F1;
        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy:
                ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");
    end;

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold
            crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ",
            100*std(testAccuracies));
    
    println(modelType, ": Average test F1 on a ", numFolds, "-fold
            crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ",
            100*std(testF1));
    
    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));
end;

"""
santaData, notSantaData = santaImagesToDatasets("BBDD/papa_noel/santa", "BBDD/papa_noel/not-a-santa");
inDS, outDS = randDataset(santaData, notSantaData);

nInstXset = convert(Int, floor(size(inDS, 1)/3));
trainset = inDS[1:nInstXset,:], outDS[1:nInstXset];
testset = inDS[(nInstXset+1):(nInstXset*2),:], outDS[(nInstXset+1):(nInstXset*2)];
validset = inDS[(nInstXset*2+1):end,:], outDS[(nInstXset*2+1):end];

trained_chain, lossestrain, lossestest, lossesvalid = entrenarClassRNA([16,8], trainset, testset, validset, 100, 0, 0.01);
plosses = plot(lossestrain)
plot!(plosses,lossestest)
plot!(plosses,lossesvalid)

using BSON: @save
@save "mymodel.bson" trained_chain

using BSON: @load
@load "mymodel.bson" trained_chain

test = imageToData("BBDD/papa_noel/santa/0.Santa.jpg");

prueba = trained_chain(test)

curvaROC = map(umbral -> (m = confusionMatrix(convert(Array{Int}, outDS), trained_chain(inDS')[1,:], umbral)[8]; (m[1,2]/sum(m[1,1:2]),m[2,2]/sum(m[2,1:2]))), 0:0.01:1);
pROC = plot(curvaROC)

accuracy, error_rate, sensitivity, specificity, pos_pred_val, neg_pred_val, F1score, confM = confusionMatrix(convert(Array{Int}, outDS), trained_chain(inDS')[1,:], 0.5);


parameters = Dict("numExecutions"=>10, "validationRatio"=>0.25,
	"topology"=>[16,8], "maxEpochs"=>100, "learningRate"=>0.01,
	"minLoss"=>0, "maxEpochsVal"=>10, "normalized"=>true, "umbral"=>0.5);
modelCrossValidation(:ANN, parameters, inDS, outDS, 10);

parameters = Dict();
parameters["maxDepth"] = 50;
modelCrossValidation(:DecisionTree, parameters, inDS, outDS, 10);

parameters = Dict();
parameters["numNeighbors"] = 5;
modelCrossValidation(:kNN, parameters, inDS, outDS, 10);

parameters = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1);
modelCrossValidation(:SVM, parameters, inDS, outDS, 10);
"""
