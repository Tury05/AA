using DelimitedFiles;
using Statistics;

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
	
end;

calculateZeroMeanNormalizationParameters = function (inputs::AbstractArray{<:Real,2})
	
end;


normalizeMinMax! = function (inputs::AbstractArray{<:Real,2},
		minMax::NTuple{2, AbstractArray{<:Real,2}})
	
end;

normalizeMinMax! = function (inputs::AbstractArray{<:Real,2})
	
end;

normalizeMinMax = function (inputs::AbstractArray{<:Real,2},
		minMax::NTuple{2, AbstractArray{<:Real,2}}=())
	
end;

normalizeMinMax = function (inputs::AbstractArray{<:Real,2})
	
end;


normalizeZeroMean! = function (inputs::AbstractArray{<:Real,2},
		meanStd::NTuple{2, AbstractArray{<:Real,2}})
	
end;

normalizeZeroMean! = function (inputs::AbstractArray{<:Real,2})
	
end;

normalizeZeroMean = function (inputs::AbstractArray{<:Real,2},
		meanStd::NTuple{2, AbstractArray{<:Real,2}})
	
end;

normalizeZeroMean = function (inputs::AbstractArray{<:Real,2})
	
end;


#3 (dificultad media)
classifyOutputs = function (outputs::AbstractArray{<:Real,2})
	
end;


#4 (dificultad media)
accuracy = function (target::AbstractArray{Bool,1},
		outputs::AbstractArray{Bool,1})
	
end;

accuracy = function (target::AbstractArray{Bool,2},
		outputs::AbstractArray{Bool,2})
	
end;

accuracy = function (target::AbstractArray{Bool,1},
		outputs::AbstractArray{<:Real,1})
	
end;

accuracy = function (target::AbstractArray{Bool,2},
		outputs::AbstractArray{<:Real,2})
	
end;


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
