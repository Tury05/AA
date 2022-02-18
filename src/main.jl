using DelimitedFiles;
using Statistics;
using Flux
using Flux.Losses

maxMinNorm = function (v, min, max)
	return (v .- min)./(max .- min);
end;

media0Norm = function (v, avg, std)
	return (v .- avg)./std;
end;

dataset = readdlm("./BBDD/iris/iris.data", ',');
f, c = size(dataset);

inDS = dataset[:, 1:c-1];
inDS[:, 1] .= 1; 
outDS = dataset[:, c];
categOutDS = unique(outDS);

@assert length(categOutDS) > 2

target = if length(categOutDS) == 2
		categOutDS[1] .== outDS
	else
		map(x -> x .== categOutDS, outDS)
	end;

maxIn = maximum(inDS, dims=1);
minIn = minimum(inDS, dims=1);
avgIn = mean(inDS, dims=1);
stdIn = std(inDS, dims=1);

indexNullColumn = last.(Tuple.(findall((maxIn .== minIn) .* (stdIn .== 0))));

if !isempty(indexNullColumn)
	c-=1;
	inDS = inDS[:, 1:end .!= indexNullColumn];
	maxIn = maxIn[:, 1:end .!= indexNullColumn];
	minIn = minIn[:, 1:end .!= indexNullColumn];
	avgIn = avgIn[:, 1:end .!= indexNullColumn];
	stdIn = stdIn[:, 1:end .!= indexNullColumn];
end;

normWithMaxMin = 0.75 .> maxMinNorm(avgIn, minIn, maxIn) .> 0.25;

inputs = Array{Float32, 2}(undef, f, c-1);

for i in 1:(c-1)
	inputs[:, i] = if(normWithMaxMin[i])
		maxMinNorm(inDS[:, i], minIn[i], maxIn[i])
	else
		media0Norm(inDS[:, i], avgIn[i], stdIn[i])
	end;
end;

#1
oneHotEncoding = function (feature::AbstractArray{<:Any,1},
		classes::AbstractArray{<:Any,1}=[])
end;

oneHotEncoding = function (feature::AbstractArray{<:Bool,1})
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
	for i in 1:size(out,2)
		inputs[:, i] = maxMinNorm(inputs[:, i], minMax[1][i], minMax[2][i])
	end
end

function normalizeMinMax!(inputs::AbstractArray{Float32,2})
	minMax = calculateMinMaxNormalizationParameters(inputs)
	for i in 1:size(out,2)
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
	for i in 1:size(out,2)
		inputs[:, i] = media0Norm(inputs[:, i], meanStd[1][i], meanStd[2][i])
	end
end

function normalizeZeroMean!(inputs::AbstractArray{Float32,2})
	meanStd = calculateZeroMeanNormalizationParameters(inputs)
	for i in 1:size(out,2)
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
		losses = cat(losses, l, dims=1);
	end;

	return (ann, losses);
	
end;

function entrenarClassRNA(topology::AbstractArray{<:Int,1},
		dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
		maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
	
	return entrenarClassRNA(topology, (first(dataset), reshape(last(dataset), :, 1)), maxEpochs, minLoss, learningRate);
end;


#6 (probar las funciones)

a = [1 2; 3 4; 5 6; 7 8]

b = [true false false; false true false; false false true; true false false]

ann = entrenarClassRNA([2,2], (a,b), 10)
