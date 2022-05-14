using Flux;
using Flux.Losses;
using Random;

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
    lossestrain = [];
    lossestest = [];
    lossesvalid = [];
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
