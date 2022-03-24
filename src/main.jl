using DelimitedFiles;
using Statistics;
using Flux;
using Flux.Losses;
using Random;

# Funciones para generar una BD con imagenes

using JLD2
using Images

# Functions that allow the conversion from images to Float64 arrays
imageToGrayArray(image:: Array{RGB{Normed{UInt8,8}},2}) = convert(Array{Float64,2}, gray.(Gray.(image)));
imageToGrayArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToGrayArray(RGB.(image));
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
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

	inputs = (first(dataset))';
	outputs = (last(dataset))';

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

		ltrain = Losses.binarycrossentropy(ann(inputs), outputs);

		push!(lossestrain, ltrain);
		push!(lossestest, ltest);
		ltest = loss(ann(first(testset)'), last(testset)');
		
		push!(lossesvalid, lvalid);
		lvalid = loss(ann(first(validset)'), last(validset)');
	
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
		dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
		testset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
		validset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
		maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01,
		maxEpochsVal::Int=20)
	
	return entrenarClassRNA(topology, (first(dataset), reshape(last(dataset), :, 1)),
		maxEpochs, minLoss, learningRate, (first(testset), reshape(last(testset), :, 1)),
		(first(validset), reshape(last(validset), :, 1)), maxEpochsVal);
end

########### PRUEBA ENTRENAMIENTO RNA ################
#<<<<<<< HEAD
inDS, outDS = readData("./BBDD/iris/iris.data")
inDS = convert(Array{Float32, 2}, inDS)
normalizeMinMax!(inDS)
outDS = oneHotEncoding(outDS)
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

function confusionMatrix(v1::AbstractArray{Bool,1}, v2::AbstractArray{Bool,1})

	vaux = v1 .== v2;
	vp = 0; vn = 0; fp = 0; fn = 0;
	
	verd = findall(vaux)
	pos = findall(v2)
	
	vp = length(findall(vaux && v2 .== 1));
	vn = length(verd) - vp;
	fp = length(pos) - vp;
	fn = length(vaux) - (vp+vn+fp);
	
	accuracy = (vn + vp)/(vn+vp+fn+fp);
	error_rate = (fn+fp)/(vn+vp+fn+fp);
	sensitivity = vp/(fp+vn);
	specificity = vn/(fp+vn);
	pos_pred_val= vp/(vp+fp);
	neg_pred_val= vn/(vn+fn);
	F1score = 2*(sensitivity * pos_pred_val / sensitivity + pos_pred_val);
	confM = [vp fp; vn fn];
	
	return (accuracy, error_rate, sensitivity, specificity, pos_pred_val, neg_pred_val, F1score, confM)
end;
			
function confusionMatrix(v1::AbstractArray{<:Real}, v2::AbstractArray{<:Real}, umbral::Real)

	vaux1 = collect(v1 .> umbral);
	vaux2 = collect(v2 .> umbral);
	confusionMatrix(vaux1, vaux2);
	
end;
	

# PRACTICA 4.2

numClasses = size(outDS, 2);
numInstances =size(outDS, 1);
rep = true;
outputs = Array{Float32,2}(undef, numInstances, numClasses);


while rep

	for numClass in 1:numClasses
		model,_ = entrenarClassRNA([8,16,8], (inDS, outDS[:, numClass]));
		global outputs[:, numClass] = model(inDS');
	end;

	vmax = maximum(outputs, dims=2);
	global outputs = (outputs .== vmax);
	print(sum(unique(outputs, dims=1), dims=1));
	global rep = any(sum(unique(outputs, dims=1), dims=1) .!= 1);
	print(rep)
end;


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
